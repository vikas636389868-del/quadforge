"""4-RoSy Cross-Field Solver for QuadForge.

Implements globally optimal direction field computation following
Knöppel et al. (2013) "Globally Optimal Direction Fields."

The field is represented as one complex number per face:
    u_f = r_f * exp(4i * θ_f)
where θ_f is the angle of one cross arm relative to a local reference
frame, and r_f is the confidence magnitude.

Key components:
  - Per-face local reference frames and parallel transport angles
  - Connection Laplacian assembly (complex-valued sparse matrix)
  - Eigensolver for the smoothest field (smallest eigenvector)
  - Feature/boundary constraint integration via penalty terms
  - Singularity detection via holonomy around vertices
  - Fallback to curvature-aligned field if eigensolver fails
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class CrossFieldData:
    """Cross-field computation result."""
    # Per-face field representation
    face_frames_e1: np.ndarray   # (F, 3) local X-axis per face
    face_frames_e2: np.ndarray   # (F, 3) local Y-axis per face
    face_normals: np.ndarray     # (F, 3) face normals
    field_angles: np.ndarray     # (F,) angle θ of cross arm in local frame
    field_complex: np.ndarray    # (F,) complex representation u = exp(4iθ)
    field_directions: np.ndarray # (F, 3) primary cross direction in 3D

    # Singularity data
    singularity_vertices: np.ndarray  # vertex indices with singularities
    singularity_indices: np.ndarray   # index (+1/4 or -1/4) per singularity


# ======================================================================
# Face geometry: normals, reference frames, adjacency
# ======================================================================

def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute unit face normals for triangles."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(cross, axis=1, keepdims=True)
    lengths[lengths < 1e-15] = 1.0
    return cross / lengths


def _compute_face_frames(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-face orthonormal reference frames (e1, e2) — vectorized.

    e1 is aligned with the first triangle edge, projected onto the face
    tangent plane and normalized. e2 = normal × e1.

    Degenerate triangles (coincident vertices, zero edge) fall back to
    an arbitrary tangent frame derived from the face normal via the
    standard "least-aligned canonical axis" trick.
    """
    if len(faces) == 0:
        return (np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64))

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    edge = v1 - v0  # (F, 3)

    # Project onto tangent plane: edge - (edge · n) n
    dots = np.einsum("ij,ij->i", edge, normals)  # (F,)
    edge_t = edge - dots[:, None] * normals      # (F, 3)

    elen = np.linalg.norm(edge_t, axis=1)        # (F,)
    degenerate = elen < 1e-15

    # For degenerate faces, build a fallback frame from the normal.
    # Pick the canonical axis least aligned with n, cross with n.
    if np.any(degenerate):
        n_deg = normals[degenerate]
        abs_n = np.abs(n_deg)
        # Axis with smallest |n[i]| is least aligned with n
        axis_idx = np.argmin(abs_n, axis=1)
        ref = np.zeros_like(n_deg)
        ref[np.arange(len(n_deg)), axis_idx] = 1.0
        fallback = np.cross(n_deg, ref)
        fallback_len = np.linalg.norm(fallback, axis=1, keepdims=True)
        fallback_len[fallback_len < 1e-15] = 1.0
        fallback /= fallback_len
        edge_t[degenerate] = fallback
        elen[degenerate] = 1.0

    e1 = edge_t / elen[:, None]
    e2 = np.cross(normals, e1)
    e2_len = np.linalg.norm(e2, axis=1, keepdims=True)
    e2_len[e2_len < 1e-15] = 1.0
    e2 /= e2_len

    return e1, e2


def _build_face_adjacency(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    """Build edge → face list adjacency from triangle faces.

    Returns dict: (min_v, max_v) → [face_indices...]
    """
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for fi in range(len(faces)):
        tri = faces[fi]
        for i in range(3):
            v0, v1 = int(tri[i]), int(tri[(i + 1) % 3])
            ek = (min(v0, v1), max(v0, v1))
            edge_to_faces.setdefault(ek, []).append(fi)
    return edge_to_faces


def _compute_corner_cotangents(
    vertices: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    """Per-triangle-corner cotangent array of shape (F, 3).

    `corner_cot[f, i]` is the cotangent of the angle at the i-th vertex
    of triangle f. For a triangle with vertices (a, b, c) the cotangent
    at vertex `a` is:
        cot(α) = ((b - a) · (c - a)) / ||(b - a) × (c - a)||

    These are the fundamental weights of the cotangent Laplacian used
    both by Knöppel's connection Laplacian and by the Poisson
    parametrization system.
    """
    va = vertices[faces[:, 0]]
    vb = vertices[faces[:, 1]]
    vc = vertices[faces[:, 2]]

    def corner(a, b, c):
        u = b - a
        v = c - a
        dot = np.einsum("ij,ij->i", u, v)
        cross = np.cross(u, v)
        sin_twice = np.linalg.norm(cross, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cot = np.where(sin_twice > 1e-15, dot / sin_twice, 0.0)
        return cot

    cot_a = corner(va, vb, vc)  # angle at vertex-index-0
    cot_b = corner(vb, vc, va)  # angle at vertex-index-1
    cot_c = corner(vc, va, vb)  # angle at vertex-index-2
    return np.stack([cot_a, cot_b, cot_c], axis=1)


def _compute_edge_data_cotangent(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_frames_e1: np.ndarray,
    face_frames_e2: np.ndarray,
    face_normals: np.ndarray,
    edge_to_faces: Dict[Tuple[int, int], List[int]],
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """Compute per-edge transport angles and cotangent weights in one pass.

    For an interior edge (u, v) shared by triangles f and g whose
    opposite (non-edge) vertices are p and q respectively, the
    cotangent Laplacian weight is:

        w_{uv} = (cot(∠p) + cot(∠q)) / 2

    where ∠p is the angle at vertex p inside triangle f (opposite to
    the edge uv), and similarly for ∠q. This is the standard discrete
    Laplace-Beltrami weight from the DEC (Discrete Exterior Calculus)
    literature and the one Knöppel 2013 prescribes for the connection
    Laplacian.

    The parallel transport angle φ is the angle by which face f's
    reference frame must rotate around the shared edge to match face
    g's reference frame, computed via the angles of the (projected)
    shared edge vector in each face's local frame.

    Returns
    -------
    transport_angles : dict edge_key → φ
    edge_weights     : dict edge_key → cotangent weight
    """
    corner_cot = _compute_corner_cotangents(vertices, faces)  # (F, 3)

    # For each interior edge, the "opposite corner" in a triangle is
    # whichever vertex is NOT on the edge. We need it to index
    # corner_cot[fi, opposite_corner].
    transport_angles: Dict[Tuple[int, int], float] = {}
    edge_weights: Dict[Tuple[int, int], float] = {}

    for edge_key, face_list in edge_to_faces.items():
        if len(face_list) != 2:
            continue
        fi, fj = face_list
        v0, v1 = edge_key

        shared_edge = vertices[v1] - vertices[v0]
        edge_len = float(np.linalg.norm(shared_edge))
        if edge_len < 1e-15:
            continue

        # Transport angle: inline of _parallel_transport_angle to avoid
        # per-edge function-call overhead for large meshes.
        n_i = face_normals[fi]
        n_j = face_normals[fj]
        e1_i = face_frames_e1[fi]
        e2_i = face_frames_e2[fi]
        e1_j = face_frames_e1[fj]
        e2_j = face_frames_e2[fj]

        edge_i = shared_edge - np.dot(shared_edge, n_i) * n_i
        len_i = float(np.linalg.norm(edge_i))
        edge_j = shared_edge - np.dot(shared_edge, n_j) * n_j
        len_j = float(np.linalg.norm(edge_j))
        if len_i < 1e-15 or len_j < 1e-15:
            phi = 0.0
        else:
            edge_i /= len_i
            edge_j /= len_j
            angle_i = math.atan2(float(np.dot(edge_i, e2_i)),
                                 float(np.dot(edge_i, e1_i)))
            angle_j = math.atan2(float(np.dot(edge_j, e2_j)),
                                 float(np.dot(edge_j, e1_j)))
            phi = angle_j - angle_i
        transport_angles[edge_key] = phi

        # Cotangent weight: find the opposite corner in each face.
        # In triangle fi = (v_a, v_b, v_c), the corner opposite edge (u, v)
        # is the one whose index is neither u nor v.
        tri_i = faces[fi]
        tri_j = faces[fj]
        cot_i = 0.0
        for k in range(3):
            vk = int(tri_i[k])
            if vk != v0 and vk != v1:
                cot_i = float(corner_cot[fi, k])
                break
        cot_j = 0.0
        for k in range(3):
            vk = int(tri_j[k])
            if vk != v0 and vk != v1:
                cot_j = float(corner_cot[fj, k])
                break

        w = 0.5 * (cot_i + cot_j)
        # Guard against obtuse-triangle negative weights (standard DEC
        # practice: clamp to a small positive). A negative cotangent
        # weight would make the connection Laplacian non-positive-
        # semidefinite and the eigensolve unstable.
        if w <= 1e-8:
            w = 1e-8
        edge_weights[edge_key] = w

    return transport_angles, edge_weights


# ======================================================================
# Parallel transport angle between adjacent faces
# ======================================================================

def _parallel_transport_angle(
    e1_a: np.ndarray, e2_a: np.ndarray, n_a: np.ndarray,
    e1_b: np.ndarray, e2_b: np.ndarray, n_b: np.ndarray,
    shared_edge: np.ndarray,
) -> float:
    """Compute the parallel transport angle φ from face A's frame to face B's.

    This is the rotation needed to align face A's reference frame with
    face B's when transporting across the shared edge.

    For a 4-RoSy field, the smoothness energy is:
        |u_A - u_B * exp(4iφ)|²
    """
    # Project shared edge onto both tangent planes
    edge_a = shared_edge - np.dot(shared_edge, n_a) * n_a
    edge_b = shared_edge - np.dot(shared_edge, n_b) * n_b

    len_a = np.linalg.norm(edge_a)
    len_b = np.linalg.norm(edge_b)

    if len_a < 1e-15 or len_b < 1e-15:
        return 0.0

    edge_a /= len_a
    edge_b /= len_b

    # Angle of shared edge in frame A
    angle_a = np.arctan2(np.dot(edge_a, e2_a), np.dot(edge_a, e1_a))
    # Angle of shared edge in frame B
    angle_b = np.arctan2(np.dot(edge_b, e2_b), np.dot(edge_b, e1_b))

    # Transport angle: how much frame A must rotate to match frame B
    return angle_b - angle_a


# ======================================================================
# Connection Laplacian assembly
# ======================================================================

def _assemble_connection_laplacian(
    num_faces: int,
    edge_to_faces: Dict[Tuple[int, int], List[int]],
    transport_angles: Dict[Tuple[int, int], float],
    edge_weights: Dict[Tuple[int, int], float],
    constraint_faces: Optional[Dict[int, float]] = None,
    constraint_strength: float = 100.0,
) -> np.ndarray:
    """Assemble the connection Laplacian as a dense complex matrix.

    The connection Laplacian L is defined as:
        L[i,i] = Σ_j w_ij  (+ constraint penalty if face i is constrained)
        L[i,j] = -w_ij * exp(4i * φ_ij)

    where w_ij is the edge weight and φ_ij is the parallel transport angle.

    The smoothest field is the eigenvector of L corresponding to the
    smallest eigenvalue (Knöppel 2013).
    """
    L = np.zeros((num_faces, num_faces), dtype=np.complex128)

    for edge_key, face_list in edge_to_faces.items():
        if len(face_list) != 2:
            continue
        fi, fj = face_list[0], face_list[1]

        phi = transport_angles.get(edge_key, 0.0)
        w = edge_weights.get(edge_key, 1.0)

        # Connection Laplacian entries (4-RoSy: multiply angle by 4)
        # Correct Knöppel 2013 formulation (matches cross_field.cpp exactly):
        #   L[fi, fj] = -w * conj(r_ij)   (transports u_fj into frame fi)
        #   L[fj, fi] = -w * r_ij         (Hermitian conjugate)
        # where r_ij = exp(4i * phi) and phi = angle_j - angle_i.
        # BUG FIX (v60): the previous code had rot and conj(rot) swapped,
        # producing the conjugate-mirrored connection Laplacian.  The matrix was
        # still Hermitian and positive semi-definite, but its smallest eigenvector
        # minimised |u_i - rot * u_j|^2 instead of |u_i - conj(rot) * u_j|^2,
        # which is the transport in the wrong direction and disagrees with the
        # C++ engine.  Fixing the swap restores consistency.
        rot = np.exp(4j * phi)

        L[fi, fi] += w
        L[fj, fj] += w
        L[fi, fj] -= w * np.conj(rot)
        L[fj, fi] -= w * rot

    # Add constraint penalties
    if constraint_faces:
        for fi, target_angle in constraint_faces.items():
            if 0 <= fi < num_faces:
                L[fi, fi] += constraint_strength

    return L


# ======================================================================
# Eigensolver (smallest eigenvector)
# ======================================================================

def _solve_smallest_eigenvector(
    L: np.ndarray,
    constraint_faces: Optional[Dict[int, complex]] = None,
    constraint_strength: float = 100.0,
    max_iterations: int = 500,
) -> np.ndarray:
    """Find the eigenvector of L with the smallest eigenvalue.

    Uses shifted inverse iteration for robustness:
      (L - σI)⁻¹ x → eigenvector for eigenvalue nearest σ
    with σ = 0 (we want the smallest eigenvalue).

    Falls back to full eigendecomposition for small matrices.
    """
    n = L.shape[0]

    if n == 0:
        return np.zeros(0, dtype=np.complex128)

    # For small matrices (< 5000 faces), use full eigendecomposition
    if n < 5000:
        # Add constraint RHS if needed
        rhs = np.zeros(n, dtype=np.complex128)
        if constraint_faces:
            for fi, target_u in constraint_faces.items():
                if 0 <= fi < n:
                    rhs[fi] = constraint_strength * target_u

        if np.any(rhs != 0):
            # Solve (L) x = rhs for constrained system
            try:
                # Add small regularisation
                L_reg = L + 1e-10 * np.eye(n, dtype=np.complex128)
                x = np.linalg.solve(L_reg, rhs)
            except np.linalg.LinAlgError:
                x = np.ones(n, dtype=np.complex128)
        else:
            # Pure eigenproblem — find smallest eigenvector
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(L)
                # eigh returns sorted eigenvalues for Hermitian matrices
                x = eigenvectors[:, 0]
            except np.linalg.LinAlgError:
                # Fallback: inverse iteration
                x = _inverse_iteration(L, max_iterations)
    else:
        # Large matrix — use inverse iteration
        rhs = np.zeros(n, dtype=np.complex128)
        if constraint_faces:
            for fi, target_u in constraint_faces.items():
                if 0 <= fi < n:
                    rhs[fi] = constraint_strength * target_u

        if np.any(rhs != 0):
            try:
                L_reg = L + 1e-8 * np.eye(n, dtype=np.complex128)
                x = np.linalg.solve(L_reg, rhs)
            except np.linalg.LinAlgError:
                x = _inverse_iteration(L, max_iterations)
        else:
            x = _inverse_iteration(L, max_iterations)

    # BUG FIX (v60): The old code divided x by its MAXIMUM component magnitude
    # (x /= max_mag), which forces the largest entry to 1.0 but leaves every
    # other component at some magnitude < 1.  A 4-RoSy field requires EVERY face
    # entry to be unit-complex (|u_fi| = 1), not just the loudest one.
    # Dividing by the global max is equivalent to a single scalar rescale, which
    # is correct for eigenvectors over the reals but wrong for complex ones where
    # each component independently encodes a direction on the unit circle.
    # Fix: normalise per-component — divide each entry by its own magnitude,
    # falling back to (1+0j) for near-zero entries (unresolved degenerate faces).
    mags = np.abs(x)
    mags_safe = np.where(mags > 1e-15, mags, 1.0)
    x = x / mags_safe

    return x


def _inverse_iteration(L: np.ndarray, max_iter: int = 300) -> np.ndarray:
    """Inverse iteration to find the eigenvector for the smallest eigenvalue."""
    n = L.shape[0]
    # Shift slightly to make matrix invertible
    sigma = 1e-8
    A = L + sigma * np.eye(n, dtype=np.complex128)

    x = np.random.RandomState(42).randn(n) + 1j * np.random.RandomState(43).randn(n)
    x /= np.linalg.norm(x)

    try:
        # Pre-factor for efficiency
        from numpy.linalg import solve
        for _ in range(max_iter):
            y = solve(A, x)
            norm = np.linalg.norm(y)
            if norm < 1e-15:
                break
            x = y / norm
    except np.linalg.LinAlgError:
        pass

    return x


# ======================================================================
# Constraint building
# ======================================================================

def _build_field_constraints(
    faces: np.ndarray,
    vertices: np.ndarray,
    face_frames_e1: np.ndarray,
    face_frames_e2: np.ndarray,
    face_normals: np.ndarray,
    edge_to_faces: Dict[Tuple[int, int], List[int]],
    feature_edges: Optional[Set[Tuple[int, int]]] = None,
    curvature_dirs: Optional[np.ndarray] = None,
) -> Tuple[Dict[int, float], Dict[int, complex]]:
    """Build constraint angles and complex targets for constrained faces.

    Returns
    -------
    constraint_angles : dict, face_idx → target angle in local frame
    constraint_complex : dict, face_idx → target complex value exp(4iθ)
    """
    constraint_angles: Dict[int, float] = {}
    constraint_complex: Dict[int, complex] = {}

    if not feature_edges:
        return constraint_angles, constraint_complex

    # Find faces adjacent to feature edges
    for edge_key, face_list in edge_to_faces.items():
        if edge_key not in feature_edges:
            continue
        v0, v1 = edge_key
        edge_vec = vertices[v1] - vertices[v0]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-15:
            continue
        edge_dir = edge_vec / edge_len

        for fi in face_list:
            if fi < 0 or fi >= len(faces):
                continue
            n = face_normals[fi]
            e1 = face_frames_e1[fi]
            e2 = face_frames_e2[fi]

            # Project edge onto face tangent plane
            edge_proj = edge_dir - np.dot(edge_dir, n) * n
            ep_len = np.linalg.norm(edge_proj)
            if ep_len < 1e-12:
                continue
            edge_proj /= ep_len

            # Angle of edge in local frame
            angle = np.arctan2(np.dot(edge_proj, e2), np.dot(edge_proj, e1))
            target_u = np.exp(4j * angle)

            # BUG FIX (v60): When a face borders multiple feature edges (e.g. a
            # mesh corner where two hard edges meet), the old code silently
            # overwrote the earlier angle/target with the latest one.  The C++
            # engine fixed this in v59 by circularly-averaging the unit-complex
            # targets.  Apply the same fix here: sum the unit-complex values and
            # re-normalise to get the geodesic midpoint on the 4-RoSy unit circle.
            if fi in constraint_complex:
                combined = constraint_complex[fi] + target_u
                m = abs(combined)
                target_u = combined / m if m > 1e-15 else target_u
                # Recover the blended angle for constraint_angles
                angle = float(np.angle(target_u)) / 4.0

            constraint_angles[fi] = angle
            constraint_complex[fi] = target_u

    return constraint_angles, constraint_complex


# ======================================================================
# Singularity detection
# ======================================================================

def _detect_singularities(
    vertices: np.ndarray,
    faces: np.ndarray,
    field_angles: np.ndarray,
    edge_to_faces: Dict[Tuple[int, int], List[int]],
    transport_angles: Dict[Tuple[int, int], float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect field singularities by computing holonomy around each vertex.

    A vertex is singular if the total rotation of the cross-field around it
    (the holonomy) is not a multiple of 2π (i.e. not zero for a 4-RoSy field).

    By the Poincaré-Hopf theorem the sum of singularity indices must equal
    the Euler characteristic χ of the surface.  For a sphere χ = 2, so the
    net count of (+1/4) minus (-1/4) singularities must equal 8.

    Implementation notes
    --------------------
    * Each face contributes to the ring of every one of its three vertices.
    * We walk the face ring CCW by following shared edges: at each step we
      pick the adjacent face that also contains vertex vi, hasn't been visited,
      and shares the correct "outgoing" edge.
    * A ring is CLOSED only if the last face is adjacent to the first face; if
      it isn't (boundary vertex), we skip the closing term.
    * We use a tighter rounding threshold (|index_q| >= 0.2) to suppress
      floating-point noise from the curvature-based fallback field.

    Returns
    -------
    sing_vertices : np.ndarray of vertex indices
    sing_indices  : np.ndarray of singularity indices (+0.25 or -0.25)
    """
    num_verts = len(vertices)

    # Build vertex → face ring
    vert_faces: Dict[int, List[int]] = {}
    for fi in range(len(faces)):
        for vi in faces[fi]:
            vert_faces.setdefault(int(vi), []).append(fi)

    # Build face ↔ face adjacency through shared edges
    face_neighbors: Dict[int, Dict[int, Tuple[int, int]]] = {}
    for edge_key, face_list in edge_to_faces.items():
        if len(face_list) != 2:
            continue
        fi, fj = face_list
        face_neighbors.setdefault(fi, {})[fj] = edge_key
        face_neighbors.setdefault(fj, {})[fi] = edge_key

    # Build edge → face lookup (faster per-edge queries)
    edge_faces = edge_to_faces  # already built

    sing_verts = []
    sing_idx = []

    for vi in range(num_verts):
        ring_faces = vert_faces.get(vi, [])
        if len(ring_faces) < 3:
            continue

        # Order faces around vi CCW by walking shared edges
        ordered = _order_face_ring_ccw(vi, ring_faces, faces, face_neighbors)
        n_ordered = len(ordered)
        if n_ordered < 3:
            continue

        # Determine whether the ring is closed (interior vertex) or open (boundary)
        first, last = ordered[0], ordered[-1]
        ring_closed = (first in face_neighbors.get(last, {}))

        steps = n_ordered if ring_closed else n_ordered - 1

        # Accumulate holonomy: for each consecutive pair of faces in the ring,
        # accumulate the residual after snapping the field-angle jump to the
        # nearest multiple of π/2 (matching for a 4-RoSy cross field).
        holonomy = 0.0
        for i in range(steps):
            fi = ordered[i]
            fj = ordered[(i + 1) % n_ordered]

            edge_key = face_neighbors.get(fi, {}).get(fj)
            if edge_key is None:
                continue

            phi = transport_angles.get(edge_key, 0.0)
            theta_i = field_angles[fi]
            theta_j = field_angles[fj]

            diff = theta_j - theta_i - phi
            # Snap to nearest multiple of π/2 (4-fold symmetry matching)
            k = round(diff * 2.0 / np.pi)
            holonomy += diff - k * np.pi / 2.0

        # Index = holonomy / (2π).  For a 4-RoSy field each singularity
        # contributes ±π/2 total holonomy = index ±1/4.
        index = holonomy / (2.0 * np.pi)
        # Round to nearest 1/4; use threshold 0.2 to avoid noise
        index_q = round(index * 4.0) / 4.0

        if abs(index_q) >= 0.2:
            sing_verts.append(vi)
            sing_idx.append(index_q)

    return np.array(sing_verts, dtype=np.int32), np.array(sing_idx, dtype=np.float64)


def _order_face_ring(
    vi: int,
    ring_faces: List[int],
    faces: np.ndarray,
    face_neighbors: Dict[int, Dict[int, Tuple[int, int]]],
) -> List[int]:
    """Legacy wrapper — use _order_face_ring_ccw for holonomy computation."""
    return _order_face_ring_ccw(vi, ring_faces, faces, face_neighbors)


def _order_face_ring_ccw(
    vi: int,
    ring_faces: List[int],
    faces: np.ndarray,
    face_neighbors: Dict[int, Dict[int, Tuple[int, int]]],
) -> List[int]:
    """Order faces around vertex vi into a (approximately) CCW ring.

    Strategy: start from any face in the ring, then repeatedly walk to an
    adjacent face that (a) also contains vi and (b) hasn't been visited yet,
    preferring the face that shares the edge OPPOSITE to the current outgoing
    half-edge around vi.

    This produces a consistent ordering sufficient for holonomy accumulation.
    If the ring is open (boundary vertex) the chain terminates when no
    unvisited neighbour that contains vi is found.
    """
    if len(ring_faces) <= 1:
        return list(ring_faces)

    ring_set = set(ring_faces)
    ordered = [ring_faces[0]]
    visited = {ring_faces[0]}

    for _ in range(len(ring_faces) - 1):
        current = ordered[-1]
        # Find the vertex in current face that comes AFTER vi (CCW direction
        # around vi means walking the face in its winding order).
        cur_face = faces[current]
        # Locate vi in the face
        vi_pos = -1
        for k, v in enumerate(cur_face):
            if int(v) == vi:
                vi_pos = k
                break
        if vi_pos < 0:
            break
        # The edge "leaving" vi in this face goes to the NEXT vertex CCW
        next_vert = int(cur_face[(vi_pos + 1) % len(cur_face)])
        edge_key = (min(vi, next_vert), max(vi, next_vert))

        # Pick the neighbour across this edge that also contains vi and
        # is still in the ring
        found_next = False
        neighbors = face_neighbors.get(current, {})
        for nb, nb_edge in neighbors.items():
            if nb in ring_set and nb not in visited:
                # Prefer the neighbour that shares the outgoing edge
                if nb_edge == edge_key:
                    ordered.append(nb)
                    visited.add(nb)
                    found_next = True
                    break
        if not found_next:
            # Fall back: any unvisited ring neighbour
            for nb in neighbors:
                if nb in ring_set and nb not in visited:
                    ordered.append(nb)
                    visited.add(nb)
                    found_next = True
                    break
        if not found_next:
            break

    return ordered


# ======================================================================
# Curvature-based fallback field
# ======================================================================

def _curvature_fallback_field(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_frames_e1: np.ndarray,
    face_frames_e2: np.ndarray,
    face_normals: np.ndarray,
    curvature_dirs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fallback: compute field from principal curvature directions or
    longest edge — fully vectorized."""
    num_faces = len(faces)
    if num_faces == 0:
        return np.zeros(0, dtype=np.float64)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    chosen_dir = np.zeros((num_faces, 3), dtype=np.float64)
    have_curv = np.zeros(num_faces, dtype=bool)

    if curvature_dirs is not None and len(curvature_dirs) >= len(vertices):
        # Average the three vertex curvature directions per face, flipping
        # the sign of any direction whose dot with the running average is
        # negative (the ±-sign ambiguity of principal directions).
        d0 = curvature_dirs[faces[:, 0]]
        d1 = curvature_dirs[faces[:, 1]]
        d2 = curvature_dirs[faces[:, 2]]
        # Flip d1 to match d0
        s01 = np.sign(np.einsum("ij,ij->i", d0, d1))
        s01[s01 == 0] = 1.0
        d1 = d1 * s01[:, None]
        avg_01 = d0 + d1
        s2 = np.sign(np.einsum("ij,ij->i", avg_01, d2))
        s2[s2 == 0] = 1.0
        d2 = d2 * s2[:, None]
        avg = d0 + d1 + d2

        # Project onto tangent plane
        dot_n = np.einsum("ij,ij->i", avg, face_normals)
        avg_t = avg - dot_n[:, None] * face_normals
        lens = np.linalg.norm(avg_t, axis=1)
        valid = lens > 1e-12
        chosen_dir[valid] = avg_t[valid] / lens[valid, None]
        have_curv[:] = valid

    # Longest-edge fallback for faces where curvature wasn't usable
    if not np.all(have_curv):
        e01 = v1 - v0
        e12 = v2 - v1
        e20 = v0 - v2
        l01 = np.linalg.norm(e01, axis=1)
        l12 = np.linalg.norm(e12, axis=1)
        l20 = np.linalg.norm(e20, axis=1)
        stack_e = np.stack([e01, e12, e20], axis=1)   # (F, 3, 3)
        stack_l = np.stack([l01, l12, l20], axis=1)   # (F, 3)
        best = np.argmax(stack_l, axis=1)             # (F,)
        longest = stack_e[np.arange(num_faces), best]  # (F, 3)

        dot_n = np.einsum("ij,ij->i", longest, face_normals)
        longest_t = longest - dot_n[:, None] * face_normals
        lens = np.linalg.norm(longest_t, axis=1)
        lens_safe = np.where(lens > 1e-12, lens, 1.0)
        longest_t = longest_t / lens_safe[:, None]

        need_fb = ~have_curv
        chosen_dir[need_fb] = longest_t[need_fb]

    # Convert the 3D tangent direction to an angle in each face's
    # local (e1, e2) frame.
    cx = np.einsum("ij,ij->i", chosen_dir, face_frames_e1)
    cy = np.einsum("ij,ij->i", chosen_dir, face_frames_e2)
    return np.arctan2(cy, cx)


# ======================================================================
# Main entry points
# ======================================================================

def compute_cross_field(
    vertices_or_he,
    faces_or_curv=None,
    feature_edges_or_feat=None,
    curvature_dirs: Optional[np.ndarray] = None,
    constraint_strength: float = 100.0,
    use_eigensolver: bool = True,
    feature_edges: Optional[Set[Tuple[int, int]]] = None,
) -> CrossFieldData:
    """Compute a globally optimal 4-RoSy cross-field on a triangle mesh.

    Supports two calling conventions:
      compute_cross_field(vertices, faces, feature_edges=..., curvature_dirs=..., ...)
      compute_cross_field(he: HalfEdgeMesh, curv: CurvatureData, feat: FeatureData, ...)

    Returns CrossFieldData.
    """
    from .halfedge import HalfEdgeMesh
    from .curvature import CurvatureData

    # --- Multi-convention dispatch ---
    if isinstance(vertices_or_he, HalfEdgeMesh):
        he = vertices_or_he
        vertices = np.asarray(he.vertices, dtype=np.float64)
        if hasattr(he, 'faces'):
            faces = np.asarray(he.faces, dtype=np.int32)
        elif hasattr(he, '_face_lists'):
            faces = np.asarray(he._face_lists, dtype=np.int32)
        else:
            raise AttributeError("HalfEdgeMesh has no 'faces' or '_face_lists' attribute")

        # faces_or_curv may be CurvatureData
        if isinstance(faces_or_curv, CurvatureData):
            curvature_dirs = faces_or_curv.dir1

        # feature_edges_or_feat may have .feature_edges
        if feature_edges_or_feat is not None and hasattr(feature_edges_or_feat, 'feature_edges'):
            feature_edges = feature_edges_or_feat.feature_edges
        elif isinstance(feature_edges_or_feat, set):
            feature_edges = feature_edges_or_feat

    else:
        vertices = np.asarray(vertices_or_he, dtype=np.float64)
        if faces_or_curv is not None:
            faces = np.asarray(faces_or_curv, dtype=np.int32)
        else:
            raise ValueError("compute_cross_field: faces required when passing raw vertices")

        # feature_edges_or_feat may be the feature_edges set
        if feature_edges_or_feat is not None and isinstance(feature_edges_or_feat, set):
            feature_edges = feature_edges_or_feat
        # feature_edges kwarg takes precedence

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    num_faces = len(faces)

    # Step 1: Face geometry
    face_normals = _compute_face_normals(vertices, faces)
    e1, e2 = _compute_face_frames(vertices, faces, face_normals)

    # Step 2: Face adjacency (edge → [face_i, face_j])
    edge_to_faces = _build_face_adjacency(faces)

    # Step 3: Transport angles + cotangent-Laplacian edge weights
    # (Knöppel 2013 — see _compute_edge_data_cotangent for details.)
    transport_angles, edge_weights = _compute_edge_data_cotangent(
        vertices, faces, e1, e2, face_normals, edge_to_faces,
    )

    # Step 4: Build constraints from feature edges
    constraint_angles, constraint_complex = _build_field_constraints(
        faces, vertices, e1, e2, face_normals,
        edge_to_faces, feature_edges, curvature_dirs,
    )

    # Step 5: Solve for the field — ALWAYS sparse. The old dense path
    # hit O(F²) memory and died on meshes above ~5K faces; the sparse
    # path is faster at every size via scipy's shift-invert Lanczos.
    if use_eigensolver and num_faces > 0:
        try:
            from .sparse_solver import (
                assemble_sparse_connection_laplacian,
                solve_sparse_eigenvector,
            )
            L = assemble_sparse_connection_laplacian(
                num_faces, edge_to_faces, transport_angles, edge_weights,
                constraint_faces=constraint_angles if constraint_angles else None,
                constraint_strength=constraint_strength,
            )
            u = solve_sparse_eigenvector(
                L,
                constraint_faces=constraint_complex if constraint_complex else None,
                constraint_strength=constraint_strength,
                num_faces=num_faces,
            )
            field_angles = np.angle(u) / 4.0
        except Exception as e:
            print(f"[QuadForge] Sparse field solver failed ({e}), "
                  f"falling back to curvature-aligned field")
            field_angles = _curvature_fallback_field(
                vertices, faces, e1, e2, face_normals, curvature_dirs,
            )
    else:
        field_angles = _curvature_fallback_field(
            vertices, faces, e1, e2, face_normals, curvature_dirs,
        )

    # Step 6: Convert angles back to 3D directions — vectorized.
    field_complex = np.exp(4j * field_angles)
    cos_t = np.cos(field_angles)[:, None]
    sin_t = np.sin(field_angles)[:, None]
    field_directions = cos_t * e1 + sin_t * e2
    fd_lens = np.linalg.norm(field_directions, axis=1, keepdims=True)
    fd_lens[fd_lens < 1e-15] = 1.0
    field_directions = field_directions / fd_lens

    # Step 7: Detect singularities
    print("[QuadForge] Detecting field singularities...")
    sing_verts, sing_indices = _detect_singularities(
        vertices, faces, field_angles, edge_to_faces, transport_angles,
    )
    if len(sing_verts) > 0:
        n_pos = np.sum(sing_indices > 0)
        n_neg = np.sum(sing_indices < 0)
        print(f"[QuadForge] Singularities: {len(sing_verts)} "
              f"(+1/4: {n_pos}, -1/4: {n_neg})")
    else:
        print("[QuadForge] No singularities detected")

    return CrossFieldData(
        face_frames_e1=e1,
        face_frames_e2=e2,
        face_normals=face_normals,
        field_angles=field_angles,
        field_complex=field_complex,
        field_directions=field_directions.astype(np.float32),
        singularity_vertices=sing_verts,
        singularity_indices=sing_indices,
    )


# ======================================================================
# Legacy compatibility wrappers
# ======================================================================

def compute_face_directions(vertices: np.ndarray, faces: list) -> np.ndarray:
    """Legacy-compatible wrapper — returns (F, 3) direction array.

    This is the API the old pipeline.py and field_smoothing.py expected.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    faces_arr = np.array(faces, dtype=np.int32)
    if faces_arr.ndim == 1:
        faces_arr = faces_arr.reshape(-1, 3)

    result = compute_cross_field(verts, faces_arr, use_eigensolver=True)
    return result.field_directions


def smooth_field(
    face_directions: np.ndarray,
    faces: list,
    vertices: np.ndarray,
    iterations: int = 5,
    strength: float = 0.5,
    feature_edges: set | None = None,
) -> np.ndarray:
    """Legacy-compatible field smoothing wrapper.

    With the new cross-field solver, smoothing is built into the global
    optimisation. This wrapper applies a few passes of local smoothing
    as a post-process refinement.
    """
    num_faces = len(faces)
    if num_faces == 0:
        return np.zeros((0, 3), dtype=np.float32)

    feature_edges = feature_edges or set()

    # Build face adjacency
    edge_map: dict = {}
    adjacency: list[set] = [set() for _ in range(num_faces)]
    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            v0, v1 = face[i], face[(i + 1) % n]
            key = (min(v0, v1), max(v0, v1))
            if key in feature_edges:
                continue
            edge_map.setdefault(key, []).append(fi)

    for face_list in edge_map.values():
        if len(face_list) == 2:
            fi, fj = face_list
            adjacency[fi].add(fj)
            adjacency[fj].add(fi)

    smoothed = np.asarray(face_directions, dtype=np.float32).copy()

    for _ in range(iterations):
        new_dirs = smoothed.copy()
        for i in range(num_faces):
            base = smoothed[i]
            s = base * (1.0 - strength)
            tw = 1.0 - strength

            for nb in adjacency[i]:
                nb_dir = smoothed[nb]
                if np.dot(base, nb_dir) < 0:
                    nb_dir = -nb_dir
                s += nb_dir * strength
                tw += strength

            if tw > 0:
                avg = s / tw
                norm = np.linalg.norm(avg)
                if norm > 1e-12:
                    new_dirs[i] = avg / norm
        smoothed = new_dirs

    return smoothed
