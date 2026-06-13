"""QuadForge — Progressive Preview System (Roadmap §13.8).

Layered preview stages so the user always sees *something* quickly and
can trust the final solve.

Three preview tiers (each builds on the previous one's cached state):

    TIER 1 — Instant (< 200 ms):
        Just the cross-field direction lines and singularity markers.
        Produced before any parametrization runs.

    TIER 2 — Medium (< 2 s):
        Approximate quad layout derived from the cross-field via a fast
        dual-contouring-like tracer. Not the final topology but good
        enough to judge edge flow and density.

    TIER 3 — Final (normal pipeline):
        Full MIQ + motorcycle-graph + cleanup. This is the real result.

Every tier writes to a ``PreviewCache`` keyed by the mesh + params
fingerprint, so a re-run with the same inputs skips straight to the
cached tier. The tiers also expose overlays (density heatmap, feature
constraints, singularity markers, risk zones) so the UI can draw them
over the preview mesh.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@dataclass
class PreviewOverlays:
    """UI overlay data associated with a preview tier."""
    field_lines: Optional[np.ndarray] = None        # (E, 2, 3) line segments
    singularity_positions: Optional[np.ndarray] = None  # (K, 3)
    density_map: Optional[np.ndarray] = None        # (V,) scalar in [0, 1]
    feature_edges: Optional[np.ndarray] = None      # (E, 2, 3) line segments
    risk_zones: Optional[np.ndarray] = None         # (F,) per-face risk in [0, 1]


@dataclass
class PreviewTier:
    tier: int                              # 1, 2, or 3
    vertices: Optional[np.ndarray] = None  # preview mesh verts (or None)
    faces: Optional[List[List[int]]] = None
    overlays: PreviewOverlays = field(default_factory=PreviewOverlays)
    elapsed_seconds: float = 0.0
    fingerprint: str = ""

    def is_mesh_preview(self) -> bool:
        return self.vertices is not None and self.faces is not None


@dataclass
class PreviewCache:
    """Keyed cache of preview tiers. Lives on the QFSettings snapshot."""
    tiers: Dict[int, PreviewTier] = field(default_factory=dict)
    fingerprint: str = ""

    def store(self, tier: PreviewTier) -> None:
        self.tiers[tier.tier] = tier

    def get(self, tier: int) -> Optional[PreviewTier]:
        return self.tiers.get(tier)

    def clear(self) -> None:
        self.tiers.clear()
        self.fingerprint = ""


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def compute_preview_fingerprint(vertices: np.ndarray,
                                faces,
                                params: dict) -> str:
    h = hashlib.sha256()
    # Only hash a downsample of the mesh — full hash is overkill for the
    # preview cache and makes the fingerprint take longer than the preview.
    stride = max(1, len(vertices) // 2048)
    h.update(vertices[::stride].tobytes())
    if isinstance(faces, np.ndarray):
        h.update(faces[::max(1, len(faces) // 2048)].tobytes())
    else:
        h.update(str(len(faces)).encode())
    for k in sorted(params.keys()):
        h.update(f"{k}={params[k]};".encode())
    return h.hexdigest()[:24]


# ---------------------------------------------------------------------------
# Tier 1 — instant field-only preview
# ---------------------------------------------------------------------------

def build_tier1_preview(vertices: np.ndarray,
                        faces: np.ndarray,
                        field_data,
                        singularities: Optional[List[int]] = None,
                        feature_edges: Optional[List[Tuple[int, int]]] = None,
                        density_map: Optional[np.ndarray] = None,
                        ) -> PreviewTier:
    """Build the instant preview.

    ``field_data`` is expected to have a ``u`` attribute of shape (V, 3)
    representing one of the four cross-field directions per vertex (the
    representative vector). It is safe to pass any object that exposes
    that attribute, including the CrossFieldData dataclass from the
    core ``field`` module, or an ad-hoc namespace with an ``u`` attribute.
    """
    t0 = time.perf_counter()
    tier = PreviewTier(tier=1)

    if field_data is not None and hasattr(field_data, "u"):
        u = np.asarray(field_data.u, dtype=np.float64)
        if u.shape[0] == len(vertices):
            # Build a short line segment per vertex along u, length = 0.5 *
            # average edge length for visual balance.
            avg_edge = _approx_avg_edge_length(vertices, faces)
            half = 0.5 * avg_edge
            starts = vertices - u * half
            ends = vertices + u * half
            segments = np.stack([starts, ends], axis=1)  # (V, 2, 3)
            tier.overlays.field_lines = segments

    if singularities:
        tier.overlays.singularity_positions = vertices[np.asarray(singularities)]

    if feature_edges:
        fe = np.asarray(feature_edges, dtype=np.int64)
        segs = np.stack([vertices[fe[:, 0]], vertices[fe[:, 1]]], axis=1)
        tier.overlays.feature_edges = segs

    if density_map is not None:
        d = np.asarray(density_map, dtype=np.float64)
        if d.size > 0:
            lo, hi = float(d.min()), float(d.max())
            if hi - lo > 1e-12:
                d = (d - lo) / (hi - lo)
            else:
                d = np.zeros_like(d)
            tier.overlays.density_map = d

    tier.elapsed_seconds = time.perf_counter() - t0
    return tier


def _approx_avg_edge_length(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(faces) == 0:
        return 1.0
    stride = max(1, len(faces) // 2000)
    sample = faces[::stride]
    lengths = []
    for tri in sample:
        a, b, c = tri[0], tri[1], tri[2]
        lengths.append(float(np.linalg.norm(vertices[b] - vertices[a])))
        lengths.append(float(np.linalg.norm(vertices[c] - vertices[b])))
        lengths.append(float(np.linalg.norm(vertices[a] - vertices[c])))
    return float(np.mean(lengths)) if lengths else 1.0


# ---------------------------------------------------------------------------
# Tier 2 — medium-quality approximate quad layout
# ---------------------------------------------------------------------------

def build_tier2_preview(vertices: np.ndarray,
                        faces: np.ndarray,
                        *,
                        target_quad_count: int = 2000,
                        sizing_field: Optional[np.ndarray] = None,
                        ) -> PreviewTier:
    """Produce a medium-quality quad approximation using a fast voxel-
    based tracer. This is NOT the final output — it's visually close
    enough to let the user cancel early if the layout is clearly wrong.

    The approach:
        1. Compute a target edge length from target_quad_count and
           mesh surface area.
        2. Rasterize the mesh into a voxel grid at that spacing.
        3. Extract a dual-contoured quad mesh from the grid.
        4. Return as-is (no smoothing, no projection).
    """
    t0 = time.perf_counter()
    tier = PreviewTier(tier=2)

    if len(vertices) == 0 or len(faces) == 0:
        tier.elapsed_seconds = time.perf_counter() - t0
        return tier

    # Estimate target edge length from area and quad count
    va = vertices[faces[:, 0]]
    vb = vertices[faces[:, 1]]
    vc = vertices[faces[:, 2]]
    area = float(0.5 * np.sum(np.linalg.norm(np.cross(vb - va, vc - va), axis=1)))
    if area <= 0.0 or target_quad_count <= 0:
        tier.elapsed_seconds = time.perf_counter() - t0
        return tier
    target_edge = float(np.sqrt(area / target_quad_count))

    # Voxelize and run a very cheap surface-nets extraction
    verts2, faces2 = _voxelize_and_extract(vertices, faces, target_edge)
    if verts2 is not None and faces2:
        tier.vertices = verts2
        tier.faces = faces2

    tier.elapsed_seconds = time.perf_counter() - t0
    return tier


def _voxelize_and_extract(vertices: np.ndarray,
                          faces: np.ndarray,
                          voxel_size: float
                          ) -> Tuple[Optional[np.ndarray], Optional[List[List[int]]]]:
    """Cheap voxel surface extraction.

    Builds a sparse set of "occupied" voxel cells that triangles pass
    through, then outputs quad faces at the shared boundary between an
    occupied and an empty neighbour cell (Surface Nets style).

    This is *not* manifold. It's for preview only.
    """
    if voxel_size <= 0:
        return None, None

    bbox_min = vertices.min(axis=0) - voxel_size
    inv = 1.0 / voxel_size

    occupied: set = set()
    # Rasterize triangle bounding boxes — cheap superset of occupancy
    for tri in faces:
        pts = vertices[tri]
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        i0, j0, k0 = np.floor((lo - bbox_min) * inv).astype(np.int64)
        i1, j1, k1 = np.floor((hi - bbox_min) * inv).astype(np.int64)
        for i in range(int(i0), int(i1) + 1):
            for j in range(int(j0), int(j1) + 1):
                for k in range(int(k0), int(k1) + 1):
                    occupied.add((i, j, k))

    if not occupied:
        return None, None

    # For every occupied cell with an empty +X / +Y / +Z neighbour, emit
    # a quad face on that boundary.
    verts_out: List[np.ndarray] = []
    faces_out: List[List[int]] = []
    vmap: Dict[Tuple[int, int, int], int] = {}

    def get_vert(key):
        idx = vmap.get(key)
        if idx is None:
            idx = len(verts_out)
            vmap[key] = idx
            pos = bbox_min + voxel_size * np.asarray(key, dtype=np.float64)
            verts_out.append(pos)
        return idx

    for cell in occupied:
        i, j, k = cell
        # +X face
        if (i + 1, j, k) not in occupied:
            q = [get_vert((i + 1, j, k)),
                 get_vert((i + 1, j + 1, k)),
                 get_vert((i + 1, j + 1, k + 1)),
                 get_vert((i + 1, j, k + 1))]
            faces_out.append(q)
        if (i - 1, j, k) not in occupied:
            q = [get_vert((i, j, k)),
                 get_vert((i, j, k + 1)),
                 get_vert((i, j + 1, k + 1)),
                 get_vert((i, j + 1, k))]
            faces_out.append(q)
        if (i, j + 1, k) not in occupied:
            q = [get_vert((i, j + 1, k)),
                 get_vert((i, j + 1, k + 1)),
                 get_vert((i + 1, j + 1, k + 1)),
                 get_vert((i + 1, j + 1, k))]
            faces_out.append(q)
        if (i, j - 1, k) not in occupied:
            q = [get_vert((i, j, k)),
                 get_vert((i + 1, j, k)),
                 get_vert((i + 1, j, k + 1)),
                 get_vert((i, j, k + 1))]
            faces_out.append(q)
        if (i, j, k + 1) not in occupied:
            q = [get_vert((i, j, k + 1)),
                 get_vert((i + 1, j, k + 1)),
                 get_vert((i + 1, j + 1, k + 1)),
                 get_vert((i, j + 1, k + 1))]
            faces_out.append(q)
        if (i, j, k - 1) not in occupied:
            q = [get_vert((i, j, k)),
                 get_vert((i, j + 1, k)),
                 get_vert((i + 1, j + 1, k)),
                 get_vert((i + 1, j, k))]
            faces_out.append(q)

    if not faces_out:
        return None, None

    return np.asarray(verts_out, dtype=np.float64), faces_out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_progressive_preview(vertices: np.ndarray,
                            faces: np.ndarray,
                            params: dict,
                            *,
                            field_data=None,
                            singularities: Optional[List[int]] = None,
                            feature_edges: Optional[List[Tuple[int, int]]] = None,
                            density_map: Optional[np.ndarray] = None,
                            cache: Optional[PreviewCache] = None,
                            want_tier: int = 2,
                            ) -> Tuple[PreviewCache, PreviewTier]:
    """Run preview stages up to ``want_tier`` and return (cache, tier).

    If the fingerprint already matches a cached entry, returns it directly.
    """
    if cache is None:
        cache = PreviewCache()

    fp = compute_preview_fingerprint(vertices, faces, params)
    if cache.fingerprint != fp:
        cache.clear()
        cache.fingerprint = fp

    # Tier 1
    if cache.get(1) is None:
        t1 = build_tier1_preview(
            vertices, faces, field_data,
            singularities=singularities,
            feature_edges=feature_edges,
            density_map=density_map,
        )
        t1.fingerprint = fp
        cache.store(t1)

    if want_tier == 1:
        return cache, cache.get(1)

    # Tier 2
    if cache.get(2) is None:
        target_qc = int(params.get("target_quad_count", 2000))
        t2 = build_tier2_preview(vertices, faces, target_quad_count=target_qc)
        t2.fingerprint = fp
        cache.store(t2)

    return cache, cache.get(want_tier) or cache.get(1)
