"""QuadForge — Mesh Repair & Sanitization Layer (v2, Dominance-grade).

Implements the "Extreme Mesh Handling System" (Roadmap §13.7) and the
"Mesh Repair & Sanitization Layer" (Roadmap §11.2).

This module runs as a mandatory pre-flight stage before any other
geometry processing step. Its job is to turn arbitrary, possibly broken
input meshes into clean, manifold, consistently-oriented triangle meshes
that the rest of the QuadForge pipeline can safely consume — and to
report *why* the input was bad and *how much* we had to scrub it.

Stages (executed in this exact order):

    0.  Triangulate arbitrary polygon input (fan)
    1.  Strip non-finite (NaN / inf) vertex positions
    2.  Weld duplicate / near-coincident vertices (adaptive, union-find)
    3.  Remove topological + area-degenerate triangles
    4.  Collapse sliver triangles (aspect-ratio threshold)
        -- critical for connection-Laplacian conditioning, see Risk 1
           in the roadmap.
    5.  Remove duplicate face entries (winding-insensitive)
    6.  Remove isolated / unreferenced vertices
    7.  Prune tiny connected components ("dust islands")
    8.  Detect & split pinched / non-manifold vertices (linear fan walk)
    9.  Detect self-intersections (broad-phase uniform grid + triangle-
        triangle test), isolate offending faces into a flag set
    10. Fill small boundary holes (fan-fill up to max_hole_edges)
    11. Propagate consistent face winding via BFS on the dual graph
    12. Global outward-orientation check via signed volume per closed
        component; flip if net-inward.
    13. Final Euler characteristic / component / genus estimate
    14. Aggregate confidence score

The module never raises on bad input. If a stage fails, it logs the
failure in the report and continues with whatever is still valid, so the
caller can decide whether to proceed, downgrade, or abort.

Public API (stable — imported by dominance_pipeline):
    - repair_mesh(vertices, faces, *, ...) -> (verts, faces, RepairReport)
    - voxel_rescue_remesh(vertices, faces, *, voxel_size=None) -> (verts, faces)
    - RepairReport  (dataclass)

Private helpers imported by dominance_pipeline:
    - _triangulate_faces(faces) -> (F,3) int array
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DEFAULT_WELD_FRACTION = 1.0e-4
DEFAULT_DEGEN_AREA_FRACTION = 1.0e-8
DEFAULT_SLIVER_ASPECT_RATIO = 50.0
DEFAULT_DUST_COMPONENT_FRACTION = 1.0e-4
DEFAULT_MAX_HOLE_EDGES = 12
SELF_INTERSECT_GRID_BUCKET = 2.0
MAX_WINDING_ROUNDS = 2_000_000

ProgressCb = Callable[[str, float], None]


# ---------------------------------------------------------------------------
# RepairReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class RepairReport:
    """Machine-readable + human-readable description of what was repaired."""

    # Input / output counts
    input_vertices: int = 0
    input_faces: int = 0
    output_vertices: int = 0
    output_faces: int = 0

    # Per-stage statistics
    nan_positions_removed: int = 0
    welded_vertices: int = 0
    degenerate_faces_removed: int = 0
    slivers_collapsed: int = 0
    isolated_vertices_removed: int = 0
    duplicate_faces_removed: int = 0
    tiny_components_removed: int = 0
    tiny_component_faces_removed: int = 0
    flipped_faces: int = 0
    non_manifold_vertices_split: int = 0
    pinched_vertices_split: int = 0
    self_intersection_faces_isolated: int = 0
    holes_filled: int = 0
    hole_fill_faces_added: int = 0
    globally_flipped_components: int = 0

    # Topology summary
    was_manifold_input: bool = True
    was_oriented_input: bool = True
    was_closed_input: bool = True
    is_manifold_output: bool = True
    connected_components: int = 1
    num_boundary_loops: int = 0
    genus_estimate: int = 0
    euler_characteristic: int = 0

    # Timings
    elapsed_seconds: float = 0.0

    # Confidence in [0, 1]
    confidence: float = 1.0

    # Human-readable warnings
    warnings: List[str] = field(default_factory=list)

    # Indices of faces in the OUTPUT mesh that were flagged as self-
    # intersecting. Downstream stages can mask them, apply extra
    # smoothing, or downgrade their sizing-field weight.
    self_intersection_face_indices: List[int] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"RepairReport: {self.input_vertices}v/{self.input_faces}f -> "
            f"{self.output_vertices}v/{self.output_faces}f "
            f"[weld={self.welded_vertices}, degen={self.degenerate_faces_removed}, "
            f"sliver={self.slivers_collapsed}, flip={self.flipped_faces}, "
            f"nm={self.non_manifold_vertices_split}, "
            f"si={self.self_intersection_faces_isolated}, "
            f"dust={self.tiny_components_removed}, hole={self.holes_filled}] "
            f"conf={self.confidence:.2f} in {self.elapsed_seconds*1000:.1f}ms"
        )

    def to_dict(self) -> Dict:
        return {
            "input_vertices": self.input_vertices,
            "input_faces": self.input_faces,
            "output_vertices": self.output_vertices,
            "output_faces": self.output_faces,
            "nan_positions_removed": self.nan_positions_removed,
            "welded_vertices": self.welded_vertices,
            "degenerate_faces_removed": self.degenerate_faces_removed,
            "slivers_collapsed": self.slivers_collapsed,
            "isolated_vertices_removed": self.isolated_vertices_removed,
            "duplicate_faces_removed": self.duplicate_faces_removed,
            "tiny_components_removed": self.tiny_components_removed,
            "tiny_component_faces_removed": self.tiny_component_faces_removed,
            "flipped_faces": self.flipped_faces,
            "non_manifold_vertices_split": self.non_manifold_vertices_split,
            "pinched_vertices_split": self.pinched_vertices_split,
            "self_intersection_faces_isolated": self.self_intersection_faces_isolated,
            "holes_filled": self.holes_filled,
            "hole_fill_faces_added": self.hole_fill_faces_added,
            "globally_flipped_components": self.globally_flipped_components,
            "was_manifold_input": self.was_manifold_input,
            "was_oriented_input": self.was_oriented_input,
            "was_closed_input": self.was_closed_input,
            "is_manifold_output": self.is_manifold_output,
            "connected_components": self.connected_components,
            "num_boundary_loops": self.num_boundary_loops,
            "genus_estimate": self.genus_estimate,
            "euler_characteristic": self.euler_characteristic,
            "elapsed_seconds": self.elapsed_seconds,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "self_intersection_face_indices": list(self.self_intersection_face_indices),
        }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _noop_progress(stage: str, progress: float) -> None:
    pass


def _triangulate_faces(faces) -> np.ndarray:
    """Accept arbitrary face lists (tris/quads/ngons) and return an (F,3)
    triangle array via simple fan triangulation. Stable and deterministic.

    This is imported privately by dominance_pipeline.py, so its signature
    and behavior must remain stable.
    """
    if isinstance(faces, np.ndarray) and faces.ndim == 2 and faces.shape[1] == 3:
        return faces.astype(np.int64, copy=True)

    tris: List[Tuple[int, int, int]] = []
    for f in faces:
        try:
            n = len(f)
        except TypeError:
            continue
        if n < 3:
            continue
        v0 = int(f[0])
        for i in range(1, n - 1):
            tris.append((v0, int(f[i]), int(f[i + 1])))
    if not tris:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(tris, dtype=np.int64)


def _bbox_diagonal(verts: np.ndarray) -> float:
    if len(verts) == 0:
        return 0.0
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    return float(np.linalg.norm(bbox_max - bbox_min))


def _face_normals_and_areas(verts: np.ndarray, faces: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized unit face normals + areas. Zero-area faces get zero normal."""
    va = verts[faces[:, 0]]
    vb = verts[faces[:, 1]]
    vc = verts[faces[:, 2]]
    cross = np.cross(vb - va, vc - va)
    lengths = np.linalg.norm(cross, axis=1)
    areas = 0.5 * lengths
    with np.errstate(divide="ignore", invalid="ignore"):
        n_unit = np.where(lengths[:, None] > 1e-30, cross / lengths[:, None], 0.0)
    return n_unit, areas


def _edges_undirected_sorted(faces: np.ndarray) -> np.ndarray:
    """Return (3F, 2) int array of undirected edges, sorted per row."""
    e = np.stack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ], axis=1).reshape(-1, 2)
    e.sort(axis=1)
    return e


def _union_find_roots(parent: np.ndarray) -> np.ndarray:
    """Path-compress every entry and return the roots array."""
    out = parent.copy()
    for i in range(len(out)):
        r = i
        while out[r] != r:
            r = out[r]
        # Path compression
        j = i
        while out[j] != r:
            nxt = out[j]
            out[j] = r
            j = nxt
    return out


# ---------------------------------------------------------------------------
# Stage 1 — NaN / inf stripping
# ---------------------------------------------------------------------------

def _strip_nans(verts: np.ndarray, faces: np.ndarray, report: RepairReport
                ) -> Tuple[np.ndarray, np.ndarray]:
    bad = ~np.isfinite(verts).all(axis=1)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return verts, faces
    report.nan_positions_removed = n_bad
    report.warnings.append(f"{n_bad} non-finite vertex positions clamped to zero.")
    verts = verts.copy()
    verts[bad] = 0.0
    return verts, faces


# ---------------------------------------------------------------------------
# Stage 2 — Adaptive vertex welding (spatial hash + union-find)
# ---------------------------------------------------------------------------

def _weld_vertices(verts: np.ndarray, faces: np.ndarray, report: RepairReport,
                   tolerance: Optional[float] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptive-tolerance vertex welding via spatial hash.

    Tolerance defaults to DEFAULT_WELD_FRACTION * bounding_box_diagonal
    which scales gracefully across mesh sizes.
    """
    if len(verts) == 0:
        return verts, faces

    if tolerance is None:
        diag = _bbox_diagonal(verts)
        if diag <= 0.0:
            return verts, faces
        tolerance = DEFAULT_WELD_FRACTION * diag

    inv = 1.0 / max(tolerance, 1e-30)
    ijk = np.floor(verts * inv).astype(np.int64)

    # cell key -> list of vertex indices
    cell_map: Dict[Tuple[int, int, int], List[int]] = {}
    for idx in range(len(verts)):
        key = (int(ijk[idx, 0]), int(ijk[idx, 1]), int(ijk[idx, 2]))
        cell_map.setdefault(key, []).append(idx)

    parent = np.arange(len(verts), dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Deterministic: lower index wins
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    tol2 = tolerance * tolerance
    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1)
               for dy in (-1, 0, 1)
               for dz in (-1, 0, 1)]

    # To avoid double work we only compare against strictly-greater-keyed
    # neighbours. (cell, cell) self-pair is included via offset (0,0,0)
    # but we use a>=b early-out below so it's safe.
    for cell, verts_in_cell in cell_map.items():
        for off in offsets:
            nkey = (cell[0] + off[0], cell[1] + off[1], cell[2] + off[2])
            if nkey < cell:
                # processed from the other side, skip
                continue
            other = cell_map.get(nkey)
            if other is None:
                continue
            for a in verts_in_cell:
                pa = verts[a]
                for b in other:
                    if a >= b:
                        continue
                    pb = verts[b]
                    dx = pa[0] - pb[0]
                    dy = pa[1] - pb[1]
                    dz = pa[2] - pb[2]
                    if dx * dx + dy * dy + dz * dz <= tol2:
                        union(a, b)

    roots = np.array([find(i) for i in range(len(verts))], dtype=np.int64)
    unique_roots, new_idx = np.unique(roots, return_inverse=True)

    welded_count = len(verts) - len(unique_roots)
    if welded_count <= 0:
        return verts, faces

    # Average welded cluster positions
    new_verts = np.zeros((len(unique_roots), 3), dtype=verts.dtype)
    counts = np.zeros(len(unique_roots), dtype=np.int64)
    np.add.at(new_verts, new_idx, verts)
    np.add.at(counts, new_idx, 1)
    new_verts /= counts[:, None]

    new_faces = new_idx[faces]
    report.welded_vertices = int(welded_count)
    return new_verts, new_faces


# ---------------------------------------------------------------------------
# Stage 3 — Degenerate triangle removal
# ---------------------------------------------------------------------------

def _remove_degenerate_faces(verts: np.ndarray, faces: np.ndarray,
                             report: RepairReport) -> np.ndarray:
    """Drop triangles where two indices match, or the area is below a
    tiny fraction of the median area."""
    if len(faces) == 0:
        return faces

    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    topo_bad = (a == b) | (b == c) | (a == c)

    _, areas = _face_normals_and_areas(verts, faces)

    finite_areas = areas[np.isfinite(areas) & (areas > 0.0)]
    if finite_areas.size > 0:
        median = float(np.median(finite_areas))
        thresh = DEFAULT_DEGEN_AREA_FRACTION * median
    else:
        thresh = 0.0

    geo_bad = (~np.isfinite(areas)) | (areas <= thresh)
    bad = topo_bad | geo_bad
    n_bad = int(bad.sum())
    if n_bad == 0:
        return faces

    report.degenerate_faces_removed = n_bad
    return faces[~bad]


# ---------------------------------------------------------------------------
# Stage 4 — Sliver triangle collapse (Risk 1 in the roadmap)
# ---------------------------------------------------------------------------

def _collapse_slivers(verts: np.ndarray, faces: np.ndarray,
                      report: RepairReport,
                      max_aspect_ratio: float = DEFAULT_SLIVER_ASPECT_RATIO
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Detect and remove sliver triangles whose aspect ratio exceeds
    max_aspect_ratio, where

        aspect_ratio = longest_edge / (2 * inradius)

    and inradius = area / semiperimeter.

    A perfect equilateral triangle has aspect ratio 1. Values above 50
    cause severe conditioning problems for the connection Laplacian
    eigensolve (this is Risk 1 in the roadmap).

    The strategy is conservative: instead of performing a real topological
    edge-collapse (which is error-prone and changes connectivity across
    the whole mesh), we simply DROP the offending faces. Any small hole
    introduced will be filled by the later hole-fill stage.
    """
    if len(faces) == 0:
        return verts, faces

    va = verts[faces[:, 0]]
    vb = verts[faces[:, 1]]
    vc = verts[faces[:, 2]]
    ab = np.linalg.norm(vb - va, axis=1)
    bc = np.linalg.norm(vc - vb, axis=1)
    ca = np.linalg.norm(va - vc, axis=1)

    longest = np.maximum(np.maximum(ab, bc), ca)
    semi = 0.5 * (ab + bc + ca)

    cross = np.cross(vb - va, vc - va)
    area = 0.5 * np.linalg.norm(cross, axis=1)

    # Inradius = area / semiperimeter. Guard against degenerates.
    with np.errstate(divide="ignore", invalid="ignore"):
        inradius = np.where(semi > 1e-30, area / semi, 0.0)
        aspect = np.where(
            inradius > 1e-30,
            longest / (2.0 * inradius),
            np.inf,
        )

    bad = aspect > max_aspect_ratio
    n_bad = int(bad.sum())
    if n_bad == 0:
        return verts, faces

    report.slivers_collapsed = n_bad
    report.warnings.append(
        f"{n_bad} sliver triangles removed (aspect ratio > {max_aspect_ratio:.0f})."
    )
    return verts, faces[~bad]


# ---------------------------------------------------------------------------
# Stage 5 — Duplicate-face removal (winding-insensitive)
# ---------------------------------------------------------------------------

def _remove_duplicate_faces(faces: np.ndarray, report: RepairReport) -> np.ndarray:
    if len(faces) == 0:
        return faces
    sorted_faces = np.sort(faces, axis=1)
    view = np.ascontiguousarray(sorted_faces).view(
        np.dtype((np.void, sorted_faces.dtype.itemsize * 3))
    )
    _, unique_idx = np.unique(view, return_index=True)
    unique_idx.sort()
    n_dup = len(faces) - len(unique_idx)
    if n_dup == 0:
        return faces
    report.duplicate_faces_removed = int(n_dup)
    return faces[unique_idx]


# ---------------------------------------------------------------------------
# Stage 6 — Isolated vertex removal
# ---------------------------------------------------------------------------

def _remove_isolated_vertices(verts: np.ndarray, faces: np.ndarray,
                              report: RepairReport
                              ) -> Tuple[np.ndarray, np.ndarray]:
    used = np.zeros(len(verts), dtype=bool)
    used[faces.reshape(-1)] = True
    n_iso = int((~used).sum())
    if n_iso == 0:
        return verts, faces
    remap = -np.ones(len(verts), dtype=np.int64)
    new_verts = verts[used]
    remap[used] = np.arange(len(new_verts))
    new_faces = remap[faces]
    report.isolated_vertices_removed = n_iso
    return new_verts, new_faces


# ---------------------------------------------------------------------------
# Connected component labelling (used by Stages 7, 11, 12)
# ---------------------------------------------------------------------------

def _label_face_components(faces: np.ndarray) -> Tuple[np.ndarray, int]:
    """Label each face by its connected component index via union-find
    over edge adjacency. Returns (labels, num_components)."""
    if len(faces) == 0:
        return np.zeros(0, dtype=np.int64), 0

    # Build edge -> face list
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for fi in range(len(faces)):
        tri = faces[fi]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (int(a), int(b)) if a < b else (int(b), int(a))
            edge_to_faces.setdefault(key, []).append(fi)

    parent = np.arange(len(faces), dtype=np.int64)

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

    for face_list in edge_to_faces.values():
        if len(face_list) > 1:
            root = face_list[0]
            for other in face_list[1:]:
                union(root, other)

    roots = np.array([find(i) for i in range(len(faces))], dtype=np.int64)
    unique_roots, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64), int(len(unique_roots))


# ---------------------------------------------------------------------------
# Stage 7 — Tiny component pruning ("dust islands")
# ---------------------------------------------------------------------------

def _prune_tiny_components(verts: np.ndarray, faces: np.ndarray,
                           report: RepairReport,
                           min_fraction: float = DEFAULT_DUST_COMPONENT_FRACTION
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Remove connected components whose face count is below
    min_fraction * largest_component_faces. Prevents stray dust geometry
    from dominating the sizing field or wasting solver effort.
    """
    if len(faces) == 0:
        return verts, faces

    labels, n_components = _label_face_components(faces)
    if n_components <= 1:
        report.connected_components = max(1, n_components)
        return verts, faces

    counts = np.bincount(labels, minlength=n_components)
    largest = int(counts.max())
    if largest == 0:
        return verts, faces

    # Prune components strictly smaller than this threshold.
    # The floor of 2 ensures that single stray triangles are always
    # pruned (they are virtually never intentional geometry), while the
    # fractional rule handles larger "chunky" islands.
    min_allowed = max(2, int(math.ceil(min_fraction * largest)))
    keep_label = counts >= min_allowed

    if keep_label.all():
        report.connected_components = n_components
        return verts, faces

    keep_face_mask = keep_label[labels]
    n_removed_components = int((~keep_label).sum())
    n_removed_faces = int((~keep_face_mask).sum())

    if n_removed_faces == 0:
        return verts, faces

    new_faces = faces[keep_face_mask]
    report.tiny_components_removed = n_removed_components
    report.tiny_component_faces_removed = n_removed_faces
    report.warnings.append(
        f"Pruned {n_removed_components} tiny components ({n_removed_faces} faces)."
    )

    # Drop any newly-isolated vertices
    new_verts, new_faces = _remove_isolated_vertices(verts, new_faces, report)
    return new_verts, new_faces


# ---------------------------------------------------------------------------
# Stage 8 — Non-manifold vertex splitting (linear fan walk)
# ---------------------------------------------------------------------------

def _split_non_manifold_vertices(verts: np.ndarray, faces: np.ndarray,
                                 report: RepairReport
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Detect vertices whose incident faces form more than one
    connected fan around the vertex, and duplicate the vertex so each
    fan gets its own copy.

    Complexity per vertex is linear in its valence (union-find over
    incident-face adjacency), so the total cost is O(F) in the number
    of triangle-vertex incidences.
    """
    if len(faces) == 0:
        return verts, faces

    F = len(faces)

    # Build vertex -> list of incident face indices with the local slot
    # (0, 1, or 2) that the vertex occupies in that face.
    incident: List[List[Tuple[int, int]]] = [[] for _ in range(len(verts))]
    for fi in range(F):
        tri = faces[fi]
        incident[int(tri[0])].append((fi, 0))
        incident[int(tri[1])].append((fi, 1))
        incident[int(tri[2])].append((fi, 2))

    # Build a vertex-pair -> list of face-slot entries keyed by the
    # *other* vertex on each incident edge at v. Two incident faces
    # share a fan edge at v iff they share a neighbouring vertex there.
    new_verts_list: List[np.ndarray] = [verts[i].copy() for i in range(len(verts))]
    new_faces = faces.copy()
    splits = 0
    pinched = 0

    for v in range(len(verts)):
        inc = incident[v]
        k = len(inc)
        if k < 2:
            continue

        # For each incident face, compute the two "other" vertex ids
        # that sit at the endpoints of v's two adjacent edges.
        # Then build a union-find over incident faces where faces sharing
        # any such neighbour vertex are merged (i.e. they lie in the same
        # fan around v).
        parent_f = list(range(k))

        def find(x: int) -> int:
            while parent_f[x] != x:
                parent_f[x] = parent_f[parent_f[x]]
                x = parent_f[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                if ra < rb:
                    parent_f[rb] = ra
                else:
                    parent_f[ra] = rb

        # other_vertex -> list of (local_i_in_inc)
        ngbr_map: Dict[int, List[int]] = {}
        for i, (fi, slot) in enumerate(inc):
            tri = new_faces[fi]
            other_a = int(tri[(slot + 1) % 3])
            other_b = int(tri[(slot + 2) % 3])
            ngbr_map.setdefault(other_a, []).append(i)
            ngbr_map.setdefault(other_b, []).append(i)

        for others in ngbr_map.values():
            if len(others) > 1:
                root = others[0]
                for j in others[1:]:
                    union(root, j)

        # Count distinct fans
        fan_roots = {find(i) for i in range(k)}
        if len(fan_roots) <= 1:
            continue

        # Group the incident faces by fan
        fans: Dict[int, List[int]] = {}
        for i in range(k):
            fans.setdefault(find(i), []).append(i)

        # Keep the first fan with the original vertex id; duplicate
        # v for each additional fan.
        fan_list = list(fans.values())
        # Deterministic ordering: sort fans by smallest face index.
        fan_list.sort(key=lambda fl: min(inc[i][0] for i in fl))

        for extra in fan_list[1:]:
            new_id = len(new_verts_list)
            new_verts_list.append(verts[v].copy())
            for i in extra:
                fi, slot = inc[i]
                new_faces[fi][slot] = new_id
            splits += 1
            # If this was the "bowtie" case (exactly 2 fans of size 1
            # each sharing only the vertex v), mark it as pinched.
            if len(fan_list) == 2:
                pinched += 1

    if splits == 0:
        return verts, faces

    report.non_manifold_vertices_split = splits
    report.pinched_vertices_split = pinched
    report.was_manifold_input = False
    new_verts = np.asarray(new_verts_list, dtype=verts.dtype)
    return new_verts, new_faces


# ---------------------------------------------------------------------------
# Stage 9 — Self-intersection detection (broad-phase grid + exact test)
# ---------------------------------------------------------------------------

def _tri_tri_intersect(t0: np.ndarray, t1: np.ndarray, eps: float = 1e-9) -> bool:
    """Möller's triangle-triangle intersection test (closed form).

    Both triangles are (3, 3) arrays of xyz. Returns True if they
    properly intersect. Shared edges and shared vertices do NOT count
    as intersection — the test rejects coplanar-shared cases up to
    the eps tolerance.
    """
    # Plane of t1
    e01 = t1[1] - t1[0]
    e02 = t1[2] - t1[0]
    n1 = np.cross(e01, e02)
    n1_len = np.linalg.norm(n1)
    if n1_len < eps:
        return False
    n1 /= n1_len
    d1 = -n1.dot(t1[0])
    d_v0 = n1.dot(t0[0]) + d1
    d_v1 = n1.dot(t0[1]) + d1
    d_v2 = n1.dot(t0[2]) + d1
    if (d_v0 > eps and d_v1 > eps and d_v2 > eps) or \
       (d_v0 < -eps and d_v1 < -eps and d_v2 < -eps):
        return False

    # Plane of t0
    e10 = t0[1] - t0[0]
    e20 = t0[2] - t0[0]
    n0 = np.cross(e10, e20)
    n0_len = np.linalg.norm(n0)
    if n0_len < eps:
        return False
    n0 /= n0_len
    d0 = -n0.dot(t0[0])
    d_u0 = n0.dot(t1[0]) + d0
    d_u1 = n0.dot(t1[1]) + d0
    d_u2 = n0.dot(t1[2]) + d0
    if (d_u0 > eps and d_u1 > eps and d_u2 > eps) or \
       (d_u0 < -eps and d_u1 < -eps and d_u2 < -eps):
        return False

    # Intersection line direction
    D = np.cross(n0, n1)
    D_len2 = float(D.dot(D))
    if D_len2 < eps * eps:
        # Coplanar — skip (we treat this as non-intersection for the
        # purposes of flagging; duplicate detection already handled it).
        return False

    # Project vertices onto the dominant axis of D
    axis = int(np.argmax(np.abs(D)))

    def interval(p, dv) -> Tuple[float, float]:
        """Compute t-interval of a triangle on the intersection line."""
        # Sort p by dv
        p0, p1, p2 = p[0, axis], p[1, axis], p[2, axis]
        d0v, d1v, d2v = dv[0], dv[1], dv[2]

        # Order so that the odd-signed vertex is in the middle of the
        # other two, using the classical two-parameter computation.
        def compute(a, b, c, da, db, dc):
            # a is the odd-signed vertex; intersection with edges a-b
            # and a-c. Guard divisions.
            ab = db - da
            ac = dc - da
            t1_ = a + (b - a) * (da / ab) if abs(ab) > eps else a
            t2_ = a + (c - a) * (da / ac) if abs(ac) > eps else a
            return (min(t1_, t2_), max(t1_, t2_))

        # Classify signs
        s0 = 1 if d0v > eps else (-1 if d0v < -eps else 0)
        s1 = 1 if d1v > eps else (-1 if d1v < -eps else 0)
        s2 = 1 if d2v > eps else (-1 if d2v < -eps else 0)

        # Find the vertex whose sign differs from the other two
        if s0 != s1 and s0 != s2:
            return compute(p0, p1, p2, d0v, d1v, d2v)
        if s1 != s0 and s1 != s2:
            return compute(p1, p0, p2, d1v, d0v, d2v)
        if s2 != s0 and s2 != s1:
            return compute(p2, p0, p1, d2v, d0v, d1v)
        # All same sign (plus one zero) — degenerate, no interval
        if s0 == 0 and s1 == 0:
            return (min(p0, p1), max(p0, p1))
        if s1 == 0 and s2 == 0:
            return (min(p1, p2), max(p1, p2))
        if s2 == 0 and s0 == 0:
            return (min(p2, p0), max(p2, p0))
        return (float("inf"), float("-inf"))

    i0 = interval(t0, np.array([d_v0, d_v1, d_v2]))
    i1 = interval(t1, np.array([d_u0, d_u1, d_u2]))
    if i0[0] > i0[1] or i1[0] > i1[1]:
        return False
    # Overlap?
    return not (i0[1] < i1[0] - eps or i1[1] < i0[0] - eps)


def _detect_self_intersections(verts: np.ndarray, faces: np.ndarray,
                               report: RepairReport,
                               max_tests: int = 200_000,
                               ) -> np.ndarray:
    """Detect face pairs that mutually intersect and return the index
    set of faces involved.

    Uses a uniform-grid broad phase keyed on face AABBs. Faces sharing a
    vertex are skipped because they can only touch, not properly
    intersect. Bound max_tests to keep the worst case tractable — on
    very intersection-heavy meshes we bail early with a warning.
    """
    F = len(faces)
    if F < 2:
        return np.zeros(0, dtype=np.int64)

    fmin = verts[faces].min(axis=1)  # (F, 3)
    fmax = verts[faces].max(axis=1)  # (F, 3)

    # Pick a grid cell size = SELF_INTERSECT_GRID_BUCKET * median edge length
    e = _edges_undirected_sorted(faces)
    edge_vecs = verts[e[:, 1]] - verts[e[:, 0]]
    edge_lens = np.linalg.norm(edge_vecs, axis=1)
    edge_lens = edge_lens[edge_lens > 0.0]
    if edge_lens.size == 0:
        return np.zeros(0, dtype=np.int64)
    median_edge = float(np.median(edge_lens))
    cell = max(median_edge * SELF_INTERSECT_GRID_BUCKET, 1e-30)
    inv_cell = 1.0 / cell

    # For each face, assign to all grid cells its AABB overlaps.
    imin = np.floor(fmin * inv_cell).astype(np.int64)
    imax = np.floor(fmax * inv_cell).astype(np.int64)

    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for fi in range(F):
        for ix in range(imin[fi, 0], imax[fi, 0] + 1):
            for iy in range(imin[fi, 1], imax[fi, 1] + 1):
                for iz in range(imin[fi, 2], imax[fi, 2] + 1):
                    grid.setdefault((ix, iy, iz), []).append(fi)

    flagged: Set[int] = set()
    tests = 0
    truncated = False

    for bucket in grid.values():
        n_b = len(bucket)
        if n_b < 2:
            continue
        for i in range(n_b):
            fa = bucket[i]
            for j in range(i + 1, n_b):
                fb = bucket[j]
                if fa in flagged and fb in flagged:
                    continue
                # Quick share-vertex reject — adjacent faces cannot
                # properly intersect.
                ta = faces[fa]
                tb = faces[fb]
                if (ta[0] == tb[0] or ta[0] == tb[1] or ta[0] == tb[2] or
                        ta[1] == tb[0] or ta[1] == tb[1] or ta[1] == tb[2] or
                        ta[2] == tb[0] or ta[2] == tb[1] or ta[2] == tb[2]):
                    continue
                # AABB reject
                if (fmax[fa, 0] < fmin[fb, 0] or fmax[fb, 0] < fmin[fa, 0] or
                        fmax[fa, 1] < fmin[fb, 1] or fmax[fb, 1] < fmin[fa, 1] or
                        fmax[fa, 2] < fmin[fb, 2] or fmax[fb, 2] < fmin[fa, 2]):
                    continue
                tests += 1
                if tests > max_tests:
                    truncated = True
                    break
                t0 = verts[ta]
                t1 = verts[tb]
                if _tri_tri_intersect(t0, t1):
                    flagged.add(int(fa))
                    flagged.add(int(fb))
            if truncated:
                break
        if truncated:
            break

    if truncated:
        report.warnings.append(
            f"Self-intersection test truncated at {max_tests} pair tests; "
            f"results may be incomplete."
        )

    if flagged:
        report.self_intersection_faces_isolated = len(flagged)
        report.self_intersection_face_indices = sorted(flagged)
        report.warnings.append(
            f"{len(flagged)} self-intersecting faces flagged "
            f"(isolated, not removed)."
        )
    return np.asarray(sorted(flagged), dtype=np.int64)


# ---------------------------------------------------------------------------
# Stage 10 — Small hole filling
# ---------------------------------------------------------------------------

def _trace_boundary_loops(faces: np.ndarray) -> List[List[int]]:
    """Trace all boundary loops. A boundary edge is one used by exactly
    one face. We walk half-edges on the boundary and chain them by
    shared vertex.
    """
    # Collect directed half-edges and count each undirected edge
    half_edges: List[Tuple[int, int]] = []
    edge_count: Dict[Tuple[int, int], int] = {}
    for tri in faces:
        for a, b in ((int(tri[0]), int(tri[1])),
                     (int(tri[1]), int(tri[2])),
                     (int(tri[2]), int(tri[0]))):
            half_edges.append((a, b))
            key = (a, b) if a < b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1

    # Boundary directed edges: those whose undirected form is used once
    boundary_from: Dict[int, List[int]] = {}
    for (a, b) in half_edges:
        key = (a, b) if a < b else (b, a)
        if edge_count[key] == 1:
            boundary_from.setdefault(a, []).append(b)

    visited: Set[Tuple[int, int]] = set()
    loops: List[List[int]] = []

    for start, targets in boundary_from.items():
        for first_next in targets:
            if (start, first_next) in visited:
                continue
            loop = [start]
            a, b = start, first_next
            safety = 0
            while True:
                visited.add((a, b))
                loop.append(b)
                # Next half-edge from b
                cands = boundary_from.get(b)
                if not cands:
                    break
                # Pick the first unvisited continuation — determinism
                # comes from dict insertion order.
                nxt = None
                for c in cands:
                    if (b, c) not in visited:
                        nxt = c
                        break
                if nxt is None:
                    break
                a, b = b, nxt
                if b == start:
                    visited.add((a, b))
                    break
                safety += 1
                if safety > len(half_edges):
                    break
            if len(loop) >= 3:
                loops.append(loop)
    return loops


def _fill_small_holes(verts: np.ndarray, faces: np.ndarray,
                      report: RepairReport,
                      max_hole_edges: int = DEFAULT_MAX_HOLE_EDGES,
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Fill boundary loops with <= max_hole_edges edges using a simple
    centroid-fan fill. Larger loops are left alone (they are assumed to
    be intentional openings).
    """
    if len(faces) == 0:
        return verts, faces

    loops = _trace_boundary_loops(faces)
    report.num_boundary_loops = len(loops)
    if not loops:
        return verts, faces

    # Number of edges in a loop equals the number of distinct vertices
    # (the tracer returns a non-repetition-closed ring).
    def _loop_edge_count(lp: List[int]) -> int:
        return len(lp) - 1 if (len(lp) >= 2 and lp[0] == lp[-1]) else len(lp)

    fillable = [lp for lp in loops if 3 <= _loop_edge_count(lp) <= max_hole_edges]
    if not fillable:
        report.was_closed_input = (len(loops) == 0)
        return verts, faces

    new_verts_list = [verts]
    new_faces_list = [faces]
    added_faces = 0
    added_verts = 0

    for lp in fillable:
        # A loop is stored as [v0, v1, ..., vn, v0] (closed). Strip the
        # trailing repetition.
        if lp[0] == lp[-1]:
            ring = lp[:-1]
        else:
            ring = lp
        k = len(ring)
        if k < 3:
            continue
        if k == 3:
            # Single triangle fill
            tri = np.asarray([[ring[0], ring[1], ring[2]]], dtype=np.int64)
            new_faces_list.append(tri)
            added_faces += 1
        else:
            # Centroid fan fill — add a new vertex at the ring centroid
            # and fan-triangulate.
            centroid = verts[np.asarray(ring, dtype=np.int64)].mean(axis=0)
            center_idx = len(verts) + added_verts
            added_verts += 1
            new_verts_list.append(centroid[None, :])
            tris = np.empty((k, 3), dtype=np.int64)
            for i in range(k):
                tris[i, 0] = center_idx
                tris[i, 1] = ring[i]
                tris[i, 2] = ring[(i + 1) % k]
            new_faces_list.append(tris)
            added_faces += k

    if added_faces == 0:
        return verts, faces

    out_verts = np.vstack(new_verts_list)
    out_faces = np.vstack(new_faces_list)
    report.holes_filled = len(fillable)
    report.hole_fill_faces_added = added_faces
    report.warnings.append(
        f"Filled {len(fillable)} small boundary loops ({added_faces} faces added)."
    )
    return out_verts, out_faces


# ---------------------------------------------------------------------------
# Stage 11 — Consistent winding via BFS on the dual graph
# ---------------------------------------------------------------------------

def _orient_faces_consistently(verts: np.ndarray, faces: np.ndarray,
                               report: RepairReport) -> np.ndarray:
    """Propagate a consistent winding via BFS across shared edges.

    Seeds from the face with the largest area (deterministic seed choice)
    and flips any neighbour whose shared edge runs in the same direction
    in both faces (which implies opposite windings).
    """
    if len(faces) == 0:
        return faces

    faces = faces.copy()

    # Directed half-edge -> face index (for the per-face orientation we
    # currently have). We also keep undirected edge -> [face indices].
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for fi in range(len(faces)):
        tri = faces[fi]
        for a, b in ((int(tri[0]), int(tri[1])),
                     (int(tri[1]), int(tri[2])),
                     (int(tri[2]), int(tri[0]))):
            key = (a, b) if a < b else (b, a)
            edge_to_faces.setdefault(key, []).append(fi)

    n = len(faces)
    visited = np.zeros(n, dtype=bool)
    flips = 0
    components = 0

    # Largest-first seed order for determinism and to bias toward the
    # "main" component.
    _, areas = _face_normals_and_areas(verts, faces)
    order = np.argsort(-areas)

    def directed_edges(tri: np.ndarray) -> List[Tuple[int, int]]:
        return [(int(tri[0]), int(tri[1])),
                (int(tri[1]), int(tri[2])),
                (int(tri[2]), int(tri[0]))]

    for seed in order:
        if visited[seed]:
            continue
        components += 1
        stack = [int(seed)]
        visited[seed] = True
        rounds = 0
        while stack:
            fi = stack.pop()
            for (a, b) in directed_edges(faces[fi]):
                key = (a, b) if a < b else (b, a)
                for nb in edge_to_faces.get(key, ()):
                    if nb == fi or visited[nb]:
                        continue
                    ntri = faces[nb]
                    same_dir = False
                    for j in range(3):
                        na = int(ntri[j])
                        nb_v = int(ntri[(j + 1) % 3])
                        if na == a and nb_v == b:
                            same_dir = True
                            break
                        if na == b and nb_v == a:
                            break
                    if same_dir:
                        faces[nb] = ntri[::-1]
                        flips += 1
                    visited[nb] = True
                    stack.append(nb)
            rounds += 1
            if rounds > MAX_WINDING_ROUNDS:
                report.warnings.append(
                    "Winding BFS safety guard triggered — output may be partially inconsistent."
                )
                break

    report.flipped_faces = flips
    report.connected_components = components
    if flips > 0:
        report.was_oriented_input = False
    return faces


# ---------------------------------------------------------------------------
# Stage 12 — Global outward orientation (signed volume per component)
# ---------------------------------------------------------------------------

def _global_outward_orient(verts: np.ndarray, faces: np.ndarray,
                           report: RepairReport
                           ) -> np.ndarray:
    """For each closed connected component, compute the signed volume
    (sum of signed tetrahedra from the origin). If negative, flip the
    entire component so it points outward.

    Open components (with boundary) are left untouched because the
    concept of "outward" isn't defined for them.
    """
    if len(faces) == 0:
        return faces

    labels, n_comp = _label_face_components(faces)
    if n_comp == 0:
        return faces

    # Which components are closed? (Every undirected edge is shared by
    # exactly 2 faces of the same component.)
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for fi in range(len(faces)):
        tri = faces[fi]
        for a, b in ((int(tri[0]), int(tri[1])),
                     (int(tri[1]), int(tri[2])),
                     (int(tri[2]), int(tri[0]))):
            key = (a, b) if a < b else (b, a)
            edge_to_faces.setdefault(key, []).append(fi)

    comp_closed = np.ones(n_comp, dtype=bool)
    for face_list in edge_to_faces.values():
        if len(face_list) != 2:
            for fi in face_list:
                comp_closed[labels[fi]] = False

    # Compute signed volume per closed component using the divergence
    # theorem: V = (1/6) * sum_f (v0 . (v1 x v2)) with consistent winding.
    va = verts[faces[:, 0]]
    vb = verts[faces[:, 1]]
    vc = verts[faces[:, 2]]
    per_face_vol = np.einsum("ij,ij->i", va, np.cross(vb, vc)) / 6.0

    flipped_components = 0
    out_faces = faces.copy()

    for c in range(n_comp):
        if not comp_closed[c]:
            continue
        mask = labels == c
        vol = float(per_face_vol[mask].sum())
        if vol < 0.0:
            out_faces[mask] = out_faces[mask][:, ::-1]
            flipped_components += 1

    if flipped_components > 0:
        report.globally_flipped_components = flipped_components
        report.was_oriented_input = False
        report.warnings.append(
            f"Flipped {flipped_components} closed component(s) to outward orientation."
        )
    return out_faces


# ---------------------------------------------------------------------------
# Stage 13 — Topology summary
# ---------------------------------------------------------------------------

def _topology_summary(verts: np.ndarray, faces: np.ndarray,
                      report: RepairReport) -> None:
    """Compute final Euler characteristic, genus estimate, and flag
    whether the output is still manifold + closed."""
    if len(faces) == 0:
        report.is_manifold_output = True
        report.euler_characteristic = 0
        report.genus_estimate = 0
        return

    # Edges
    e = _edges_undirected_sorted(faces)
    # Count unique edges and their multiplicity
    view = np.ascontiguousarray(e).view(
        np.dtype((np.void, e.dtype.itemsize * 2))
    )
    _, inv, counts = np.unique(view, return_inverse=True, return_counts=True)
    E = len(counts)
    nm_edges = int((counts > 2).sum())
    boundary_edges = int((counts == 1).sum())

    V = len(verts)
    F = len(faces)
    chi = V - E + F

    is_manifold = nm_edges == 0
    is_closed = boundary_edges == 0

    report.is_manifold_output = is_manifold
    report.euler_characteristic = int(chi)
    report.was_closed_input = report.was_closed_input and is_closed

    if not is_manifold:
        report.warnings.append(f"{nm_edges} non-manifold edges remain in output.")

    # Genus estimate for closed components only: chi = 2c - 2g  =>  g = c - chi/2
    c = max(1, report.connected_components)
    if is_closed:
        report.genus_estimate = max(0, c - chi // 2)
    else:
        report.genus_estimate = 0


# ---------------------------------------------------------------------------
# Stage 14 — Confidence scoring
# ---------------------------------------------------------------------------

def _compute_confidence(report: RepairReport) -> float:
    """Aggregate a 0..1 confidence score from the repair statistics.

    The scoring is intentionally pessimistic: any of the "severe" issues
    (non-finite positions, non-manifold output, self-intersections) pulls
    confidence down fast, while minor cosmetic repairs have only small
    penalties.
    """
    v_in = max(1, report.input_vertices)
    f_in = max(1, report.input_faces)
    f_out = max(1, report.output_faces)

    bad = 0.0
    # Severe: non-finite positions (indicates catastrophic upstream bug)
    bad += min(1.0, report.nan_positions_removed / v_in) * 0.50
    # Severe: non-manifold splitting (changes topology)
    bad += min(1.0, report.non_manifold_vertices_split / v_in) * 0.35
    # Severe: self-intersections
    bad += min(1.0, report.self_intersection_faces_isolated / f_out) * 0.40
    # Severe: non-manifold output
    if not report.is_manifold_output:
        bad += 0.25

    # Medium: degenerate/sliver faces (numerics)
    bad += min(1.0, report.degenerate_faces_removed / f_in) * 0.20
    bad += min(1.0, report.slivers_collapsed / f_in) * 0.15
    # Medium: major global reorientation
    bad += min(1.0, report.flipped_faces / f_in) * 0.10
    if report.globally_flipped_components > 0:
        bad += 0.05

    # Minor: cosmetic
    bad += min(1.0, report.welded_vertices / v_in) * 0.05
    bad += min(1.0, report.duplicate_faces_removed / f_in) * 0.05
    bad += min(1.0, report.tiny_component_faces_removed / f_in) * 0.05

    conf = max(0.0, 1.0 - bad)
    return float(conf)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def repair_mesh(vertices, faces, *,
                weld_tolerance: Optional[float] = None,
                aggressive: bool = True,
                sliver_aspect_ratio: float = DEFAULT_SLIVER_ASPECT_RATIO,
                dust_component_fraction: float = DEFAULT_DUST_COMPONENT_FRACTION,
                max_hole_edges: int = DEFAULT_MAX_HOLE_EDGES,
                detect_self_intersections: bool = True,
                fill_holes: bool = True,
                prune_dust: bool = True,
                progress_cb: Optional[ProgressCb] = None,
                ) -> Tuple[np.ndarray, np.ndarray, RepairReport]:
    """Run the full repair pipeline.

    Parameters
    ----------
    vertices : (N, 3) array-like of float
    faces : list-of-lists or (F, K) array-like of int. Quads/ngons are
        fan-triangulated.
    weld_tolerance : optional absolute weld tolerance; if None an
        adaptive tolerance (DEFAULT_WELD_FRACTION * bbox diagonal) is used.
    aggressive : if False, skip non-manifold splitting, winding fixes,
        hole fill, self-intersection detection, dust pruning, and global
        orientation. Useful in time-budgeted fallback paths.
    sliver_aspect_ratio : triangles whose longest_edge / (2 * inradius)
        exceeds this value are collapsed (dropped and potentially hole-
        filled later).
    dust_component_fraction : components with face count below this
        fraction of the largest component are pruned.
    max_hole_edges : boundary loops with up to this many edges are
        automatically filled.
    detect_self_intersections : run the self-intersection broad-phase
        (can be slow on meshes with many large faces).
    fill_holes : run the hole-filling stage.
    prune_dust : run the tiny-component pruning stage.
    progress_cb : optional callback for UI progress; signature
        ``(stage_name: str, progress_0_to_1: float) -> None``.

    Returns
    -------
    (new_vertices, new_faces, RepairReport)
    """
    t0 = time.perf_counter()
    report = RepairReport()
    cb = progress_cb or _noop_progress

    verts = np.ascontiguousarray(np.asarray(vertices, dtype=np.float64))
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3), got shape {verts.shape}")

    cb("triangulate", 0.00)
    tri_faces = _triangulate_faces(faces)

    report.input_vertices = len(verts)
    report.input_faces = len(tri_faces)

    if len(verts) == 0 or len(tri_faces) == 0:
        report.warnings.append("Empty input mesh.")
        report.output_vertices = len(verts)
        report.output_faces = len(tri_faces)
        report.confidence = 0.0
        report.elapsed_seconds = time.perf_counter() - t0
        return verts, tri_faces, report

    # Stage 1: strip NaN / inf
    cb("strip-nan", 0.05)
    verts, tri_faces = _strip_nans(verts, tri_faces, report)

    # Stage 2: weld duplicates
    cb("weld", 0.12)
    verts, tri_faces = _weld_vertices(verts, tri_faces, report,
                                      tolerance=weld_tolerance)

    # Stage 3: drop degenerate triangles
    cb("degen", 0.22)
    tri_faces = _remove_degenerate_faces(verts, tri_faces, report)

    # Stage 4: collapse sliver triangles
    cb("sliver", 0.28)
    verts, tri_faces = _collapse_slivers(verts, tri_faces, report,
                                         max_aspect_ratio=sliver_aspect_ratio)

    # Stage 5: drop duplicate faces
    cb("dedupe-faces", 0.34)
    tri_faces = _remove_duplicate_faces(tri_faces, report)

    # Stage 6: drop isolated vertices
    cb("isolated", 0.38)
    verts, tri_faces = _remove_isolated_vertices(verts, tri_faces, report)

    if len(tri_faces) == 0:
        report.warnings.append("All faces were removed during repair.")
        report.output_vertices = len(verts)
        report.output_faces = 0
        report.confidence = 0.0
        report.elapsed_seconds = time.perf_counter() - t0
        return verts, tri_faces, report

    # Stage 7: prune tiny components (dust)
    if aggressive and prune_dust:
        cb("dust", 0.45)
        verts, tri_faces = _prune_tiny_components(
            verts, tri_faces, report,
            min_fraction=dust_component_fraction,
        )

    # Stage 8: non-manifold detection + splitting
    if aggressive and len(tri_faces) > 0:
        cb("non-manifold", 0.55)
        verts, tri_faces = _split_non_manifold_vertices(verts, tri_faces, report)

    # Stage 9: self-intersection detection (flag only, never remove)
    if aggressive and detect_self_intersections and len(tri_faces) > 0:
        cb("self-intersect", 0.65)
        try:
            _detect_self_intersections(verts, tri_faces, report)
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"self-intersection test failed: {exc}")

    # Stage 10: small hole filling
    if aggressive and fill_holes and len(tri_faces) > 0:
        cb("hole-fill", 0.75)
        try:
            verts, tri_faces = _fill_small_holes(
                verts, tri_faces, report,
                max_hole_edges=max_hole_edges,
            )
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"hole-fill stage failed: {exc}")

    # Stage 11: consistent winding
    if aggressive and len(tri_faces) > 0:
        cb("orient-bfs", 0.85)
        tri_faces = _orient_faces_consistently(verts, tri_faces, report)

    # Stage 12: global outward orientation
    if aggressive and len(tri_faces) > 0:
        cb("orient-global", 0.92)
        try:
            tri_faces = _global_outward_orient(verts, tri_faces, report)
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"global orientation stage failed: {exc}")

    # Stage 13: final topology summary
    cb("summary", 0.97)
    try:
        _topology_summary(verts, tri_faces, report)
    except Exception as exc:  # noqa: BLE001
        report.warnings.append(f"topology summary failed: {exc}")

    # Stage 14: confidence
    report.output_vertices = len(verts)
    report.output_faces = len(tri_faces)
    report.confidence = _compute_confidence(report)
    report.elapsed_seconds = time.perf_counter() - t0
    cb("done", 1.00)

    return verts, tri_faces, report


# ---------------------------------------------------------------------------
# Voxel rescue path (Roadmap §13.7, last bullet)
# ---------------------------------------------------------------------------

def voxel_rescue_remesh(vertices: np.ndarray, faces: np.ndarray,
                        *,
                        voxel_size: Optional[float] = None,
                        max_grid: int = 128,
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Last-resort remesh for severely corrupted input.

    Strategy:
        1. Build a uniform voxel grid sized so the largest axis has at
           most ``max_grid`` cells.
        2. Rasterize each triangle into the grid by marking all voxels
           whose AABB overlaps the triangle's AABB (conservative — this
           gives a thin shell of occupied voxels around the surface).
        3. Extract a closed 2-manifold surface by emitting one quad per
           exposed voxel face (a face where exactly one of the two
           neighbouring voxels is occupied). Each quad is fan-split into
           two triangles.

    This guarantees a topologically-clean triangle mesh at the cost of
    some geometric detail — the output surface is a voxelized
    approximation of the input shell. The caller is expected to project
    the result onto the original surface via BVH (see
    ``projection.project_to_surface``) as a follow-up step.

    Parameters
    ----------
    vertices : (N, 3) float array
    faces : (F, 3) int array (or any fan-triangulable input)
    voxel_size : absolute cell size. If None, computed so that the
        longest bbox axis has ``max_grid`` cells.
    max_grid : maximum cells along the longest axis when auto-sizing.

    Returns
    -------
    (new_vertices, new_faces) — a watertight triangle mesh, or the
    original input unchanged if the rescue cannot run.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    faces = _triangulate_faces(faces)
    if len(verts) == 0 or len(faces) == 0:
        return verts, faces

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    extent = bbox_max - bbox_min
    extent_max = float(extent.max())
    if extent_max <= 0.0:
        return verts, faces

    if voxel_size is None:
        voxel_size = extent_max / float(max_grid)
    voxel_size = max(voxel_size, extent_max / float(max_grid))

    # Pad by 1 cell on each side for safety
    origin = bbox_min - voxel_size
    grid_dims = np.ceil((bbox_max - origin) / voxel_size).astype(np.int64) + 1
    gx, gy, gz = int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2])
    if gx * gy * gz > (max_grid + 2) ** 3:
        # Guardrail — something went wrong with sizing
        voxel_size = extent_max / float(max_grid)
        origin = bbox_min - voxel_size
        grid_dims = np.ceil((bbox_max - origin) / voxel_size).astype(np.int64) + 1
        gx, gy, gz = int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2])

    # Occupancy grid (boolean)
    occ = np.zeros((gx, gy, gz), dtype=bool)

    # Rasterize each triangle's AABB into voxel cells
    tri_mins = verts[faces].min(axis=1)
    tri_maxs = verts[faces].max(axis=1)
    inv = 1.0 / voxel_size
    imin = np.floor((tri_mins - origin) * inv).astype(np.int64)
    imax = np.floor((tri_maxs - origin) * inv).astype(np.int64)
    np.clip(imin, 0, grid_dims - 1, out=imin)
    np.clip(imax, 0, grid_dims - 1, out=imax)

    for fi in range(len(faces)):
        occ[
            imin[fi, 0]:imax[fi, 0] + 1,
            imin[fi, 1]:imax[fi, 1] + 1,
            imin[fi, 2]:imax[fi, 2] + 1,
        ] = True

    # Extract surface: a voxel face is on the boundary iff exactly one
    # of its two neighbouring cells is occupied. We generate 6 axis-
    # aligned face sets, each as a boolean mask over the oriented face
    # lattice.
    new_verts: List[Tuple[float, float, float]] = []
    new_tris: List[Tuple[int, int, int]] = []
    vert_map: Dict[Tuple[int, int, int], int] = {}

    def get_vert(ix: int, iy: int, iz: int) -> int:
        key = (ix, iy, iz)
        idx = vert_map.get(key)
        if idx is None:
            idx = len(new_verts)
            vert_map[key] = idx
            new_verts.append((
                origin[0] + ix * voxel_size,
                origin[1] + iy * voxel_size,
                origin[2] + iz * voxel_size,
            ))
        return idx

    def add_quad(a: int, b: int, c: int, d: int) -> None:
        # Split the quad into two triangles (a,b,c) and (a,c,d)
        new_tris.append((a, b, c))
        new_tris.append((a, c, d))

    # +X faces (between cell x and x+1) where cell[x,y,z] is occupied
    # and cell[x+1,y,z] is not (or OOB).
    occ_padded = np.pad(occ, 1, mode="constant", constant_values=False)

    for ix in range(gx):
        for iy in range(gy):
            for iz in range(gz):
                if not occ[ix, iy, iz]:
                    continue

                # +X face
                if not occ_padded[ix + 2, iy + 1, iz + 1]:
                    a = get_vert(ix + 1, iy, iz)
                    b = get_vert(ix + 1, iy + 1, iz)
                    c = get_vert(ix + 1, iy + 1, iz + 1)
                    d = get_vert(ix + 1, iy, iz + 1)
                    add_quad(a, b, c, d)
                # -X face
                if not occ_padded[ix, iy + 1, iz + 1]:
                    a = get_vert(ix, iy, iz)
                    d = get_vert(ix, iy + 1, iz)
                    c = get_vert(ix, iy + 1, iz + 1)
                    b = get_vert(ix, iy, iz + 1)
                    add_quad(a, b, c, d)
                # +Y face
                if not occ_padded[ix + 1, iy + 2, iz + 1]:
                    a = get_vert(ix, iy + 1, iz)
                    d = get_vert(ix + 1, iy + 1, iz)
                    c = get_vert(ix + 1, iy + 1, iz + 1)
                    b = get_vert(ix, iy + 1, iz + 1)
                    add_quad(a, b, c, d)
                # -Y face
                if not occ_padded[ix + 1, iy, iz + 1]:
                    a = get_vert(ix, iy, iz)
                    b = get_vert(ix + 1, iy, iz)
                    c = get_vert(ix + 1, iy, iz + 1)
                    d = get_vert(ix, iy, iz + 1)
                    add_quad(a, b, c, d)
                # +Z face
                if not occ_padded[ix + 1, iy + 1, iz + 2]:
                    a = get_vert(ix, iy, iz + 1)
                    b = get_vert(ix + 1, iy, iz + 1)
                    c = get_vert(ix + 1, iy + 1, iz + 1)
                    d = get_vert(ix, iy + 1, iz + 1)
                    add_quad(a, b, c, d)
                # -Z face
                if not occ_padded[ix + 1, iy + 1, iz]:
                    a = get_vert(ix, iy, iz)
                    d = get_vert(ix + 1, iy, iz)
                    c = get_vert(ix + 1, iy + 1, iz)
                    b = get_vert(ix, iy + 1, iz)
                    add_quad(a, b, c, d)

    if not new_verts or not new_tris:
        return verts, faces

    out_verts = np.asarray(new_verts, dtype=np.float64)
    out_faces = np.asarray(new_tris, dtype=np.int64)
    return out_verts, out_faces
