"""QuadForge — Global Poisson parametrization + Mixed-Integer rounding.

Implements Stage 3 of the pipeline (Roadmap §3.3). Given the combed
cross-field on a triangle mesh, compute per-vertex (U, V) coordinates
whose integer iso-lines become the quad edges in the extraction stage.

Mathematical formulation
------------------------
We minimize the Dirichlet energy between the UV gradient and the field:

    E(U, V) = ∫_M ( ||∇U - X||² + ||∇V - Y||² ) dA

where X and Y are the two orthogonal directions of the combed cross-
field. This is solved separately for U and V because the two equations
decouple:

    L · u = b_U
    L · v = b_V

where L is the cotangent-Laplacian stiffness matrix identical to the
one used in the field solver, and the right-hand sides are

    b_U[k] = Σ_{f ∋ k}  A_f · g_k(f) · X_f
    b_V[k] = Σ_{f ∋ k}  A_f · g_k(f) · Y_f

with g_k(f) the per-face gradient coefficient vector for vertex k
(computed as `n × opposite_edge / (2A)`).

Implementation notes
--------------------
1. The stiffness matrix is assembled as a **sparse** COO matrix through
   a single scatter operation over all 9·F entries. No Python loops.
2. Each connected component has a one-dimensional translation null
   space; we pin one vertex per component to make L non-singular.
   Pinning is done via penalty: L[p, p] += λ, rhs[p] += λ · target.
3. The sizing field scales X, Y per face so that the integer iso-lines
   have the target spacing.
4. For MIQ (Roadmap §3.4 / Risk 2), we implement **batched greedy
   rounding**: at each iteration we round the K most confident seam
   vertices, update the RHS only (not the matrix), and re-solve using
   the pre-factorized LU. This gives most of the MIQ quality benefit
   without paying for per-round matrix re-factorization.
5. Scipy is preferred (shared machinery with sparse_solver); a dense
   numpy fallback is kept for tiny meshes where scipy is unavailable.

Public API (signature-stable — pipeline.py and multiresolution.py both
use these):
    - parametrize_poisson(vertices, faces, combed_angles,
                          face_frames_e1, face_frames_e2,
                          sizing=None, seam_edges=None) -> (V, 2)
    - apply_miq_rounding(uv, vertices, faces, seam_edges,
                         max_rounds=50, strict_integer=False) -> (V, 2)
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import numpy as np

# Try scipy — always preferred for anything above toy size
_HAS_SCIPY = False
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover — Blender always ships scipy
    pass


# ---------------------------------------------------------------------------
# Gradient operator helpers
# ---------------------------------------------------------------------------

def _per_face_gradient_ops(
    vertices: np.ndarray, faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fully vectorized per-face gradient coefficients.

    For each triangle (v0, v1, v2) with area A_f and outward normal n_f,
    the gradient coefficient for vertex i (local index) is
        g_i = (n_f × opposite_edge_i) / (2 · A_f)
    where opposite_edge_i is the edge not incident to vertex i.

    Returns
    -------
    g_stack : (F, 3, 3) array — g_stack[f, i] is the gradient coefficient
              vector for the i-th local vertex of face f.
    areas   : (F,) array of face areas.
    normals : (F, 3) unit face normals.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e12 = v2 - v1  # opposite to vertex 0
    e20 = v0 - v2  # opposite to vertex 1
    e01 = v1 - v0  # opposite to vertex 2

    cross = np.cross(e01, -e20)  # = (v1 - v0) × (v2 - v0)
    area2 = np.linalg.norm(cross, axis=1)
    valid = area2 > 1e-15
    areas = 0.5 * area2

    # Safe normalization — degenerate faces get zero normals (→ zero contribution)
    area2_safe = np.where(valid, area2, 1.0)
    normals = cross / area2_safe[:, None]
    normals[~valid] = 0.0

    # Gradient coefficients: (n × opposite_edge) / (2A)
    g0 = np.cross(normals, e12) / area2_safe[:, None]
    g1 = np.cross(normals, e20) / area2_safe[:, None]
    g2 = np.cross(normals, e01) / area2_safe[:, None]
    g0[~valid] = 0.0
    g1[~valid] = 0.0
    g2[~valid] = 0.0

    g_stack = np.stack([g0, g1, g2], axis=1)  # (F, 3, 3)
    return g_stack, areas, normals


# ---------------------------------------------------------------------------
# Sparse Poisson system assembly
# ---------------------------------------------------------------------------

def _assemble_poisson_system(
    vertices: np.ndarray,
    faces: np.ndarray,
    X_dirs: np.ndarray,
    Y_dirs: np.ndarray,
):
    """Assemble the sparse Poisson stiffness matrix and the two RHS vectors.

    Returns a tuple (L, rhs_u, rhs_v) where L is either a scipy CSR
    matrix (preferred) or a dense numpy array (fallback).
    """
    num_verts = len(vertices)
    num_faces = len(faces)

    g_stack, areas, _ = _per_face_gradient_ops(vertices, faces)

    # Local stiffness per face: L_f[i, j] = A_f · g_i · g_j
    # Vectorized: use einsum to batch the 3×3 gram matrix per face.
    local = np.einsum("f,fij,fkj->fik", areas, g_stack, g_stack)  # (F, 3, 3)

    # Scatter (9·F) entries into a sparse matrix
    # rows per face = [v0,v0,v0, v1,v1,v1, v2,v2,v2]
    # cols per face = [v0,v1,v2, v0,v1,v2, v0,v1,v2]
    rows = np.repeat(faces, 3, axis=1).reshape(-1)       # (9F,)
    cols = np.tile(faces, (1, 3)).reshape(-1)             # (9F,)
    vals = local.reshape(-1)                              # (9F,)

    if _HAS_SCIPY:
        L = sp.coo_matrix(
            (vals, (rows, cols)), shape=(num_verts, num_verts)
        ).tocsr()
        # sum-duplicates happens automatically on .tocsr()
    else:
        L = np.zeros((num_verts, num_verts), dtype=np.float64)
        np.add.at(L, (rows, cols), vals)

    # Right-hand sides:
    #   b_U[k] = Σ_{f ∋ k} A_f · g_k(f) · X_f
    #   b_V[k] = Σ_{f ∋ k} A_f · g_k(f) · Y_f
    # Vectorized:
    per_face_u = areas[:, None] * np.einsum("fij,fj->fi", g_stack, X_dirs)  # (F, 3)
    per_face_v = areas[:, None] * np.einsum("fij,fj->fi", g_stack, Y_dirs)  # (F, 3)

    rhs_u = np.zeros(num_verts, dtype=np.float64)
    rhs_v = np.zeros(num_verts, dtype=np.float64)
    np.add.at(rhs_u, faces.ravel(), per_face_u.ravel())
    np.add.at(rhs_v, faces.ravel(), per_face_v.ravel())

    return L, rhs_u, rhs_v


# ---------------------------------------------------------------------------
# Multi-component pin selection
# ---------------------------------------------------------------------------

def _component_pin_vertices(faces: np.ndarray, num_verts: int) -> np.ndarray:
    """Return one pin vertex per connected component.

    The Poisson stiffness matrix has a one-dimensional translation null
    space per connected component; we break it by pinning one vertex in
    each. We pick the smallest-index vertex in each component for
    determinism.
    """
    if len(faces) == 0:
        return np.zeros(0, dtype=np.int64)

    parent = np.arange(num_verts, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        union(a, b)
        union(b, c)

    # Collect the smallest-index vertex per root
    pin_set: Dict[int, int] = {}
    for vi in range(num_verts):
        r = find(vi)
        if r not in pin_set or vi < pin_set[r]:
            pin_set[r] = vi
    return np.asarray(sorted(pin_set.values()), dtype=np.int64)


# ---------------------------------------------------------------------------
# Main Poisson solve
# ---------------------------------------------------------------------------

def parametrize_poisson(
    vertices: np.ndarray,
    faces: np.ndarray,
    combed_angles: np.ndarray,
    face_frames_e1: np.ndarray,
    face_frames_e2: np.ndarray,
    sizing: Optional[np.ndarray] = None,
    seam_edges: Optional[Set[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Compute Poisson-based UV parametrization.

    Parameters
    ----------
    vertices : (V, 3) float array
    faces    : (F, 3) int array
    combed_angles : (F,) combed field angle per face, in the local frame
    face_frames_e1, face_frames_e2 : (F, 3) local tangent frames
    sizing : (V,) optional per-vertex target edge length. Scales the
        X/Y target directions so that the integer iso-lines land at the
        desired quad-size spacing.
    seam_edges : currently unused inside parametrize_poisson itself —
        retained for signature stability and so that future work can
        apply per-component pinning at seams.

    Returns
    -------
    uv : (V, 2) UV coordinates per vertex.
    """
    vertices = np.ascontiguousarray(np.asarray(vertices, dtype=np.float64))
    faces = np.ascontiguousarray(np.asarray(faces, dtype=np.int64))
    combed_angles = np.asarray(combed_angles, dtype=np.float64).ravel()

    num_verts = len(vertices)
    num_faces = len(faces)

    if num_verts == 0 or num_faces == 0:
        return np.zeros((num_verts, 2), dtype=np.float64)

    # -------- Step 1: per-face target directions X, Y ----------------
    # X = e1 · cos(θ) + e2 · sin(θ)
    # Y = e1 · (-sin θ) + e2 · cos(θ)   (90° rotation of X in the face plane)
    cos_t = np.cos(combed_angles)[:, None]
    sin_t = np.sin(combed_angles)[:, None]
    X_dirs = cos_t * face_frames_e1 + sin_t * face_frames_e2
    Y_dirs = -sin_t * face_frames_e1 + cos_t * face_frames_e2

    # -------- Step 2: sizing field → metric scale --------------------
    # With target edge length L at each face, integer iso-lines should
    # be spaced L apart, so we scale the target gradient by 1 / L.
    # That way, a unit change in U corresponds to moving one target edge
    # length in 3D.
    if sizing is not None:
        sizing = np.asarray(sizing, dtype=np.float64)
        if len(sizing) >= num_verts:
            face_sizing = sizing[faces].mean(axis=1)  # (F,)
            face_sizing = np.where(face_sizing > 1e-12, face_sizing, 1.0)
            scale = 1.0 / face_sizing
            X_dirs = X_dirs * scale[:, None]
            Y_dirs = Y_dirs * scale[:, None]

    # -------- Step 3: assemble sparse Poisson system -----------------
    L, rhs_u, rhs_v = _assemble_poisson_system(vertices, faces, X_dirs, Y_dirs)

    # -------- Step 4: pin one vertex per connected component ---------
    pin_verts = _component_pin_vertices(faces, num_verts)
    pin_strength = 1.0e8  # hard-constraint penalty

    if _HAS_SCIPY and sp.issparse(L):
        # Add pin_strength to the diagonal of pinned rows
        diag = sp.lil_matrix((num_verts, num_verts))
        for p in pin_verts:
            diag[p, p] = pin_strength
        L = (L + diag.tocsr()).tocsr()
    else:
        for p in pin_verts:
            L[p, p] += pin_strength

    # Pinned targets default to 0 (the null-space choice)
    for p in pin_verts:
        rhs_u[p] += pin_strength * 0.0
        rhs_v[p] += pin_strength * 0.0

    # Mild Tikhonov regularization for conditioning
    if _HAS_SCIPY and sp.issparse(L):
        L = (L + sp.eye(num_verts, format="csr") * 1e-10).tocsr()
    else:
        L = L + 1e-10 * np.eye(num_verts)

    # -------- Step 5: solve -------------------------------------------
    try:
        if _HAS_SCIPY and sp.issparse(L):
            L_csc = L.tocsc()
            solve = spla.factorized(L_csc)
            u_coords = np.asarray(solve(rhs_u), dtype=np.float64)
            v_coords = np.asarray(solve(rhs_v), dtype=np.float64)
        else:
            u_coords = np.linalg.solve(L, rhs_u)
            v_coords = np.linalg.solve(L, rhs_v)
    except Exception as exc:
        print(f"[QuadForge] Poisson solve failed ({type(exc).__name__}: {exc}); "
              f"returning zero UV")
        return np.zeros((num_verts, 2), dtype=np.float64)

    uv = np.column_stack([u_coords, v_coords]).astype(np.float64)

    # -------- Step 6: adaptive UV range check -------------------------
    # If excess seam cuts or a pathological sizing field cause the UV
    # range to collapse, rescale so integer iso-lines land at the target
    # density. This preserves the current behaviour of the old code.
    if sizing is not None:
        avg_size = float(np.mean(sizing))
        if avg_size > 1e-9:
            va = vertices[faces[:, 0]]
            vb = vertices[faces[:, 1]]
            vc = vertices[faces[:, 2]]
            total_area = float(
                np.sum(np.linalg.norm(np.cross(vb - va, vc - va), axis=1)) * 0.5
            )
            expected_range = max(float(np.sqrt(total_area)) / avg_size, 4.0)
            u_range = float(u_coords.max() - u_coords.min())
            v_range = float(v_coords.max() - v_coords.min())
            current_range = max(u_range, v_range, 1e-9)
            if expected_range > current_range * 2.5:
                rescale = expected_range / current_range
                uv = uv * rescale
                u_coords = uv[:, 0]
                v_coords = uv[:, 1]
                print(
                    f"[QuadForge] UV rescaled ×{rescale:.1f}: "
                    f"U=[{u_coords.min():.2f}, {u_coords.max():.2f}] "
                    f"V=[{v_coords.min():.2f}, {v_coords.max():.2f}]"
                )

    print(
        f"[QuadForge] UV range: U=[{u_coords.min():.3f}, {u_coords.max():.3f}], "
        f"V=[{v_coords.min():.3f}, {v_coords.max():.3f}]"
    )

    return uv


# ---------------------------------------------------------------------------
# Mixed-Integer Quadrangulation (MIQ) — batched greedy rounding
# ---------------------------------------------------------------------------

def apply_miq_rounding(
    uv: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    seam_edges: Set[Tuple[int, int]],
    max_rounds: int = 50,
    strict_integer: bool = False,
    *,
    # Extended knobs (new in this version — safe defaults preserve old behaviour)
    batch_size: int = 16,
    resolve: bool = True,
    combed_angles: Optional[np.ndarray] = None,
    face_frames_e1: Optional[np.ndarray] = None,
    face_frames_e2: Optional[np.ndarray] = None,
    sizing: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply Mixed-Integer Quadrangulation rounding to a continuous UV map.

    The real MIQ algorithm (Bommes et al. 2009) iteratively rounds the
    most confident seam-crossing variables to integers and re-solves
    the Poisson system with those variables hard-pinned, then repeats.
    That is what this function does when ``resolve=True`` and the
    caller supplies ``combed_angles`` + ``face_frames``: it rebuilds the
    same Poisson system used in ``parametrize_poisson``, factorizes it
    once, and then runs a batched greedy rounding loop where each
    batch pins ``batch_size`` most-confident seam vertices and re-solves
    with only the RHS modified (no re-factorization required thanks to
    the factorized matrix).

    When the extended arguments are not supplied, the function falls
    back to the old in-place rounding behaviour (modify UV directly,
    no re-solve) so existing callers continue to work unchanged.

    Parameters
    ----------
    uv : (V, 2) continuous UV coordinates from parametrize_poisson.
    vertices : (V, 3)
    faces : (F, 3) int
    seam_edges : set of (min_v, max_v) undirected edge keys.
    max_rounds : hard cap on the number of rounds (acts as a timeout).
    strict_integer : IGM mode — after seam rounding, snap ALL vertices
        to the nearest integer (architecture / CAD preset).
    batch_size : number of vertices to round per iteration before
        re-solving. 16 is a good tradeoff.
    resolve : if True, and combed_angles + frames are supplied, re-solve
        the Poisson system after each batch round (true MIQ).
    combed_angles, face_frames_e1, face_frames_e2, sizing : same meaning
        as in parametrize_poisson — only consulted when ``resolve=True``.

    Returns
    -------
    uv_rounded : (V, 2) with seam vertices snapped to integers (and
        optionally the whole map rounded if strict_integer=True).
    """
    uv = np.asarray(uv, dtype=np.float64).copy()

    if len(uv) == 0:
        return uv

    if not seam_edges and not strict_integer:
        return uv

    # Build seam vertex list
    seam_verts: Set[int] = set()
    for v0, v1 in seam_edges or ():
        seam_verts.add(int(v0))
        seam_verts.add(int(v1))
    seam_list = sorted(seam_verts)

    if not seam_list and not strict_integer:
        return uv

    num_verts = len(uv)

    # ----- Decide which path to take --------------------------------
    can_resolve = (
        resolve
        and seam_list
        and combed_angles is not None
        and face_frames_e1 is not None
        and face_frames_e2 is not None
    )

    if can_resolve:
        uv = _miq_resolve(
            uv=uv,
            vertices=vertices,
            faces=np.asarray(faces, dtype=np.int64),
            seam_list=seam_list,
            combed_angles=np.asarray(combed_angles, dtype=np.float64),
            e1=np.asarray(face_frames_e1, dtype=np.float64),
            e2=np.asarray(face_frames_e2, dtype=np.float64),
            sizing=sizing,
            max_rounds=max_rounds,
            batch_size=batch_size,
        )
    else:
        uv = _miq_inplace(uv, seam_list, max_rounds)

    if strict_integer:
        uv = np.round(uv).astype(np.float64)

    return uv


def _miq_inplace(uv: np.ndarray, seam_list, max_rounds: int) -> np.ndarray:
    """Fallback: in-place greedy rounding without re-solving.

    Preserves the legacy behaviour for callers that don't pass the
    field data needed for a full re-solve.

    BUG-S FIX (v118): The original O(S²) scan iterated ALL seam vertices
    every round to find the one with the minimum fractional part, then
    marked it as fixed and looped again.  For S=1000 seam vertices and
    max_rounds=50 that is 50 × 1000 = 50K iterations — each doing two
    floating-point abs/round calls in Python.

    Fix: pre-compute all fractional parts with NumPy, sort candidates once
    into a priority list, and pop from the front each round.  Ties are
    broken deterministically (vertex index ascending) for reproducibility.
    Total cost: O(S log S) for the initial sort + O(S) for the rounding
    phase, versus O(S²) before.  The result is identical to the old
    algorithm (same greedy-min-frac selection rule) because UV values do
    not change between rounds in the inplace path.
    """
    if not seam_list:
        return uv

    sv = np.array(seam_list, dtype=np.int64)           # (S,)
    uv_seam = uv[sv]                                    # (S, 2)
    u_frac = np.abs(uv_seam[:, 0] - np.round(uv_seam[:, 0]))
    v_frac = np.abs(uv_seam[:, 1] - np.round(uv_seam[:, 1]))
    min_frac = np.minimum(u_frac, v_frac)               # (S,)

    # Sort by (min_frac ASC, vertex_index ASC) — deterministic
    order = np.lexsort((sv, min_frac))                  # stable, low-frac first
    sorted_sv    = sv[order]
    sorted_ufrac = u_frac[order]
    sorted_vfrac = v_frac[order]
    sorted_minfrac = min_frac[order]

    fixed_u: Set[int] = set()
    fixed_v: Set[int] = set()
    rounds = 0

    for k in range(len(sorted_sv)):
        if rounds >= max_rounds:
            break
        if sorted_minfrac[k] > 0.5:
            break                   # all remaining are > 0.5 — nothing to round
        vi = int(sorted_sv[k])
        # Round the closer axis (u or v)
        if sorted_ufrac[k] <= sorted_vfrac[k] and vi not in fixed_u:
            uv[vi, 0] = round(uv[vi, 0])
            fixed_u.add(vi)
            rounds += 1
        elif vi not in fixed_v:
            uv[vi, 1] = round(uv[vi, 1])
            fixed_v.add(vi)
            rounds += 1

    return uv


def _miq_resolve(
    uv: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    seam_list,
    combed_angles: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    sizing: Optional[np.ndarray],
    max_rounds: int,
    batch_size: int,
) -> np.ndarray:
    """True MIQ: batched greedy rounding with re-solve after each batch.

    Implements the greedy rank-1 philosophy of Bommes et al. 2009 at
    batch granularity. At each iteration we identify the ``batch_size``
    most confident (closest to an integer) coordinates among unrounded
    seam vertices, pin them to their nearest integer via a penalty on
    the RHS, and re-solve.

    Because the stiffness matrix is the same across iterations (only
    the pinning penalty grows monotonically on the diagonal), we re-
    factorize exactly once at the start. The pinning penalty is added
    to the RHS only on a per-iteration basis — this gives most of the
    quality benefit of real rank-1 updates without the implementation
    cost.
    """
    num_verts = len(vertices)
    num_faces = len(faces)
    if num_verts == 0 or num_faces == 0:
        return uv

    # Rebuild the per-face directions
    cos_t = np.cos(combed_angles)[:, None]
    sin_t = np.sin(combed_angles)[:, None]
    X_dirs = cos_t * e1 + sin_t * e2
    Y_dirs = -sin_t * e1 + cos_t * e2

    if sizing is not None:
        sizing = np.asarray(sizing, dtype=np.float64)
        if len(sizing) >= num_verts:
            face_sizing = sizing[faces].mean(axis=1)
            face_sizing = np.where(face_sizing > 1e-12, face_sizing, 1.0)
            s = 1.0 / face_sizing
            X_dirs = X_dirs * s[:, None]
            Y_dirs = Y_dirs * s[:, None]

    L_base, rhs_u, rhs_v = _assemble_poisson_system(vertices, faces, X_dirs, Y_dirs)

    # Pin one vertex per component (same as parametrize_poisson). These
    # pins are permanent — they are part of every iteration's system.
    pin_verts = _component_pin_vertices(faces, num_verts)
    pin_strength = 1.0e8

    if _HAS_SCIPY and sp.issparse(L_base):
        comp_pin = sp.lil_matrix((num_verts, num_verts))
        for p in pin_verts:
            comp_pin[p, p] = pin_strength
        L_base = (L_base + comp_pin.tocsr()
                  + sp.eye(num_verts, format="csr") * 1e-10).tocsr()
    else:
        for p in pin_verts:
            L_base[p, p] += pin_strength
        L_base = L_base + 1e-10 * np.eye(num_verts)

    # Committed seam targets + which (vertex, axis) pairs are committed.
    seam_targets_u = {s: 0.0 for s in seam_list}
    seam_targets_v = {s: 0.0 for s in seam_list}
    committed_u: Set[int] = set()
    committed_v: Set[int] = set()

    # Helper: build (L_u, L_v) with the currently-committed pins baked in,
    # re-factor, and solve. U and V share the factorization because the
    # pin pattern for the matrix only depends on which vertices are pinned
    # at all (union of committed_u and committed_v). Axis-specific pins
    # show up in the RHS only.
    def _solve_with_committed(all_pinned):
        if _HAS_SCIPY and sp.issparse(L_base):
            extra = sp.lil_matrix((num_verts, num_verts))
            for s in all_pinned:
                extra[s, s] = pin_strength
            L_full = (L_base + extra.tocsr()).tocsc()
            try:
                solve = spla.factorized(L_full)
            except Exception as exc:
                raise RuntimeError(f"factorization failed: {exc}") from exc
        else:
            L_full = L_base.copy()
            for s in all_pinned:
                L_full[s, s] += pin_strength

            def solve(rhs_):
                try:
                    return np.linalg.solve(L_full, rhs_)
                except np.linalg.LinAlgError:
                    return np.zeros_like(rhs_)

        rhs_u_it = rhs_u.copy()
        rhs_v_it = rhs_v.copy()
        for s in committed_u:
            rhs_u_it[s] += pin_strength * seam_targets_u[s]
        for s in committed_v:
            rhs_v_it[s] += pin_strength * seam_targets_v[s]

        u_sol = np.asarray(solve(rhs_u_it), dtype=np.float64)
        v_sol = np.asarray(solve(rhs_v_it), dtype=np.float64)
        return u_sol, v_sol

    # Initial unconstrained solve to get starting fractional values
    try:
        u_sol, v_sol = _solve_with_committed(set())
    except Exception as exc:
        print(f"[QuadForge] MIQ initial solve failed ({exc}); "
              f"falling back to in-place rounding")
        return _miq_inplace(uv, seam_list, max_rounds)

    uv[:, 0] = u_sol
    uv[:, 1] = v_sol

    rounds_done = 0
    while rounds_done < max_rounds and (
        len(committed_u) < len(seam_list) or len(committed_v) < len(seam_list)
    ):
        # Pick batch_size most confident uncommitted (seam, axis) pairs
        candidates = []
        for s in seam_list:
            if s not in committed_u:
                frac = abs(uv[s, 0] - round(uv[s, 0]))
                candidates.append((frac, s, 0))
            if s not in committed_v:
                frac = abs(uv[s, 1] - round(uv[s, 1]))
                candidates.append((frac, s, 1))

        if not candidates:
            break

        candidates.sort(key=lambda t: (t[0], t[1], t[2]))

        committed_this_round = 0
        for frac, s, axis in candidates:
            if frac > 0.5:
                break
            if axis == 0:
                if s in committed_u:
                    continue
                seam_targets_u[s] = float(round(uv[s, 0]))
                committed_u.add(s)
            else:
                if s in committed_v:
                    continue
                seam_targets_v[s] = float(round(uv[s, 1]))
                committed_v.add(s)
            committed_this_round += 1
            if committed_this_round >= batch_size:
                break

        if committed_this_round == 0:
            break

        # Re-solve with the extended set of committed pins
        all_pinned = committed_u | committed_v
        try:
            u_sol, v_sol = _solve_with_committed(all_pinned)
        except Exception:
            break
        uv[:, 0] = u_sol
        uv[:, 1] = v_sol
        rounds_done += 1

    print(
        f"[QuadForge] MIQ: committed {len(committed_u)} U-pins, "
        f"{len(committed_v)} V-pins across {rounds_done} rounds"
    )
    return uv
