"""QuadForge v1.1 — Incremental / Selection Remeshing.

Remeshes only the selected faces of a mesh while preserving the rest.
The selected region is extracted, fed through the QuadForge pipeline,
and the result is stitched back to the original mesh by matching
boundary vertices via BVH nearest-point lookup.

Public API
----------
extract_selection(vertices, faces, selected_mask)
    → SelectionData

run_selection_pipeline(sel_data, params, progress_cb=None)
    → (remeshed_verts, remeshed_faces)

stitch_selection(original_verts, original_faces, selected_mask,
                 remeshed_verts, remeshed_faces,
                 boundary_snap_distance=None)
    → (new_verts, new_faces)

Algorithm (Overview)
--------------------
1.  Extract the selected face sub-mesh, keeping a map from sub-mesh
    vertex indices back to original vertex indices.

2.  Identify "boundary vertices" — vertices that are shared by both
    selected and unselected faces.  Their world-space positions must
    be preserved in the output.

3.  Run the standard QuadForge pipeline on the sub-mesh.  Boundary
    vertices are injected as hard feature constraints so the field and
    parametrization respect the boundary shape.

4.  After the pipeline, find the output boundary (free-edge vertices),
    and snap each one to the nearest original boundary vertex using a
    BVH k-d tree.  Any output boundary vertex within `snap_distance`
    of an original boundary vertex is merged to it.

5.  Re-assemble:
      • Remove all originally-selected faces from the original mesh.
      • Compact (remove now-orphaned vertices).
      • Append the remeshed vertices (already snapped at boundary).
      • Append the remeshed faces (index-shifted).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set, Tuple

from .spatial import KDTree


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SelectionData:
    """Extracted sub-mesh data for selection remeshing."""

    # Sub-mesh geometry
    sub_vertices: np.ndarray          # (V_sub, 3) float64
    sub_faces: np.ndarray             # (F_sub, 3) int32 — triangulated

    # Index maps
    orig_to_sub: np.ndarray           # (V_orig,) int32  — -1 if not in sub-mesh
    sub_to_orig: np.ndarray           # (V_sub,) int32

    # Boundary
    boundary_sub_indices: Set[int]    # sub-mesh vertex indices on the boundary
    boundary_orig_indices: Set[int]   # original mesh vertex indices on boundary

    # Masks
    selected_face_mask: np.ndarray    # (F_orig,) bool

    # Optional per-vertex attributes (forwarded from input mesh)
    sub_normals:        Optional[np.ndarray] = None
    sub_vertex_colors:  Optional[np.ndarray] = None
    sub_material_ids:   Optional[np.ndarray] = None


@dataclass
class StitchResult:
    """Result of stitching a remeshed region back into the original mesh."""
    vertices: np.ndarray    # (V_new, 3) float64
    faces: List[List[int]]  # mixed quads + tris

    # Stats
    boundary_snapped: int = 0        # vertices snapped to original boundary
    boundary_total:   int = 0
    snap_distance_used: float = 0.0


# ---------------------------------------------------------------------------
# Step 1 — Extract selected region
# ---------------------------------------------------------------------------

def extract_selection(
    vertices:      np.ndarray,     # (V, 3) float64
    faces:         np.ndarray,     # (F, 3) int32 — must be triangulated
    selected_mask: np.ndarray,     # (F,) bool
    normals:       Optional[np.ndarray] = None,
    vertex_colors: Optional[np.ndarray] = None,
    material_ids:  Optional[np.ndarray] = None,
) -> SelectionData:
    """Extract the selected face region into a standalone sub-mesh.

    Parameters
    ----------
    vertices      : (V, 3) mesh vertex positions
    faces         : (F, 3) triangulated face indices
    selected_mask : (F,)   bool, True = selected
    normals       : (V, 3) optional per-vertex normals
    vertex_colors : (V, 3) optional per-vertex RGB colors
    material_ids  : (F,)   optional per-face material IDs

    Returns
    -------
    SelectionData with sub-mesh geometry and index maps.
    """
    vertices  = np.asarray(vertices,  dtype=np.float64)
    faces     = np.asarray(faces,     dtype=np.int32)
    selected  = np.asarray(selected_mask, dtype=bool)

    if selected.sum() == 0:
        raise ValueError("No faces selected.")
    if selected.sum() == len(faces):
        raise ValueError("All faces selected — use regular remesh instead.")

    # --- Find which original vertices are referenced by selected faces ------
    sel_faces    = faces[selected]          # (F_sel, 3)
    unsel_faces  = faces[~selected]         # (F_unsel, 3)

    used_orig = np.unique(sel_faces)        # sorted array of orig vertex indices

    # --- Build index maps ----------------------------------------------------
    orig_to_sub = np.full(len(vertices), -1, dtype=np.int32)
    orig_to_sub[used_orig] = np.arange(len(used_orig), dtype=np.int32)
    sub_to_orig = used_orig.copy()

    # --- Remap face indices --------------------------------------------------
    sub_faces = orig_to_sub[sel_faces]     # (F_sel, 3) — sub-mesh indices

    sub_vertices = vertices[used_orig]    # (V_sub, 3)

    # --- Identify boundary vertices -----------------------------------------
    #   A vertex is a boundary vertex if it appears in both selected and
    #   unselected faces.
    used_unsel = set(unsel_faces.ravel().tolist())
    used_sel   = set(used_orig.tolist())
    boundary_orig = used_sel & used_unsel
    boundary_sub  = {int(orig_to_sub[v]) for v in boundary_orig}

    # --- Optional per-vertex attributes -------------------------------------
    sub_normals = vertices[used_orig]  # placeholder; replaced below
    sub_normals = None
    if normals is not None:
        sub_normals = np.asarray(normals, dtype=np.float64)[used_orig]

    sub_vcolors = None
    if vertex_colors is not None:
        sub_vcolors = np.asarray(vertex_colors, dtype=np.float32)[used_orig]

    sub_mats = None
    if material_ids is not None:
        sub_mats = np.asarray(material_ids, dtype=np.int32)[selected]

    return SelectionData(
        sub_vertices=sub_vertices,
        sub_faces=sub_faces,
        orig_to_sub=orig_to_sub,
        sub_to_orig=sub_to_orig,
        boundary_sub_indices=boundary_sub,
        boundary_orig_indices=boundary_orig,
        selected_face_mask=selected,
        sub_normals=sub_normals,
        sub_vertex_colors=sub_vcolors,
        sub_material_ids=sub_mats,
    )


# ---------------------------------------------------------------------------
# Step 2 — Run the pipeline on the sub-mesh
# ---------------------------------------------------------------------------

def run_selection_pipeline(
    sel_data:    SelectionData,
    params,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Run the QuadForge pipeline on the extracted selection sub-mesh.

    Boundary vertices are injected as hard feature constraints so the
    field respects the boundary shape.  This prevents the parametrization
    from running iso-lines through the boundary in ways that would produce
    T-junctions at the stitch seam.

    Parameters
    ----------
    sel_data    : output of extract_selection()
    params      : QFParams-like namespace
    progress_cb : optional progress callback (stage_name, 0–1)

    Returns
    -------
    (remeshed_verts, remeshed_faces) — faces are lists of int
    """
    from .bridge import remesh as bridge_remesh

    # Build a minimal mesh-like object for the pipeline
    class _SubMesh:
        pass

    mesh = _SubMesh()
    mesh.vertices      = sel_data.sub_vertices
    mesh.faces         = sel_data.sub_faces
    mesh.uv_seam_edges = set()
    mesh._features     = None
    mesh.uv_coords     = None
    mesh.vertex_colors = sel_data.sub_vertex_colors
    mesh.material_ids  = sel_data.sub_material_ids
    mesh.face_smooth   = None

    # Inject boundary vertices as pre-computed feature edges so the
    # cross-field aligns to the selection boundary loop.
    boundary_edges: Set[Tuple[int, int]] = set()
    sub_faces = sel_data.sub_faces
    for fi, face in enumerate(sub_faces):
        for i in range(len(face)):
            v0 = int(face[i])
            v1 = int(face[(i + 1) % len(face)])
            if v0 in sel_data.boundary_sub_indices and v1 in sel_data.boundary_sub_indices:
                boundary_edges.add((min(v0, v1), max(v0, v1)))

    # Attach pre-detected feature edges so pipeline Stage 2 can use them
    if boundary_edges:
        class _Features:
            def __init__(self, edges):
                self.feature_edges = frozenset(edges)
                self.all_features  = frozenset(edges)
        mesh._features = _Features(boundary_edges)

    result = bridge_remesh(mesh, params, progress_cb=progress_cb)

    # Support both _PipelineResult and legacy (verts, faces) tuple
    if hasattr(result, 'verts'):
        return result.verts, result.faces
    return result


def _noop_progress(stage: str, progress: float):
    pass


# ---------------------------------------------------------------------------
# Step 3 — Stitch the remeshed region back into the original mesh
# ---------------------------------------------------------------------------

def stitch_selection(
    original_verts:  np.ndarray,         # (V_orig, 3)
    original_faces:  np.ndarray,         # (F_orig, 3) triangulated
    selected_mask:   np.ndarray,         # (F_orig,) bool
    remeshed_verts:  np.ndarray,         # (V_rem,  3) output of pipeline
    remeshed_faces:  List[List[int]],    # output of pipeline
    sel_data:        SelectionData,
    boundary_snap_distance: Optional[float] = None,
) -> StitchResult:
    """Stitch the remeshed region back into the original mesh.

    Algorithm
    ---------
    1.  Build a BVH over the original boundary vertices.
    2.  For each remeshed vertex that lies near the boundary, snap it
        to the nearest original boundary vertex (within snap_distance).
    3.  Remove the selected faces from the original mesh.
    4.  Remove vertices that become orphaned after face removal.
    5.  Append the (snapped) remeshed vertices.
    6.  Append the remeshed faces (with shifted indices).
    7.  Return the combined mesh.

    Parameters
    ----------
    original_verts          : (V_orig, 3) float64
    original_faces          : (F_orig, 3) int32
    selected_mask           : (F_orig,) bool
    remeshed_verts          : (V_rem, 3)  pipeline output
    remeshed_faces          : list of index lists  pipeline output
    sel_data                : SelectionData from extract_selection()
    boundary_snap_distance  : max distance for boundary vertex snapping.
                              None → auto (2 × mean remeshed edge length)
    """
    original_verts = np.asarray(original_verts, dtype=np.float64)
    original_faces = np.asarray(original_faces, dtype=np.int32)
    remeshed_verts = np.asarray(remeshed_verts, dtype=np.float64)
    selected       = np.asarray(selected_mask,  dtype=bool)

    # --- 1. Auto-compute snap distance ---------------------------------------
    if boundary_snap_distance is None:
        if len(remeshed_faces) > 0 and len(remeshed_verts) > 0:
            edge_lengths = []
            for f in remeshed_faces[:min(100, len(remeshed_faces))]:
                for i in range(len(f)):
                    v0 = remeshed_verts[f[i]]
                    v1 = remeshed_verts[f[(i + 1) % len(f)]]
                    edge_lengths.append(float(np.linalg.norm(v1 - v0)))
            boundary_snap_distance = 2.0 * float(np.mean(edge_lengths)) if edge_lengths else 0.1
        else:
            boundary_snap_distance = 0.1

    # --- 2. BVH over original boundary vertices ------------------------------
    bnd_orig_indices = list(sel_data.boundary_orig_indices)
    bnd_orig_positions = original_verts[bnd_orig_indices]  # (B, 3)

    # Build a simple k-d tree for boundary lookup
    kd = KDTree(bnd_orig_positions)

    # --- 3. Detect remeshed boundary vertices and snap them ------------------
    # Remeshed boundary = vertices that are on the free edge of the remeshed mesh.
    rem_boundary = _find_boundary_vertices_from_faces(len(remeshed_verts), remeshed_faces)

    # Work on a mutable copy of remeshed verts
    snapped_verts = remeshed_verts.copy()
    snap_remap    = np.arange(len(remeshed_verts), dtype=np.int32)  # identity initially

    snapped_count = 0
    # Map: original vertex index → position in the "keep" original vertex list
    # (built later, but we need to track which orig boundary verts get re-used)
    orig_bound_used: dict[int, int] = {}  # orig_idx → will be assigned an index after compaction

    for rem_vi in rem_boundary:
        pos = snapped_verts[rem_vi]
        dist, nn_idx = kd.query(pos.flatten())
        dist = float(dist[0])
        if dist <= boundary_snap_distance:
            orig_vi = bnd_orig_indices[int(nn_idx[0])]
            # Snap position to original boundary vertex
            snapped_verts[rem_vi] = original_verts[orig_vi]
            orig_bound_used[orig_vi] = rem_vi   # track which rem vertex represents this orig vert
            snapped_count += 1

    # --- 4. Build the "kept" original mesh (unselected faces only) -----------
    unsel_faces = original_faces[~selected]        # (F_unsel, 3)
    used_orig_verts_set = set(unsel_faces.ravel().tolist())

    # Compact: map old vertex indices to new compact indices
    used_orig_sorted = sorted(used_orig_verts_set)
    old_to_new_orig  = np.full(len(original_verts), -1, dtype=np.int32)
    for new_i, old_i in enumerate(used_orig_sorted):
        old_to_new_orig[old_i] = new_i

    kept_verts = original_verts[used_orig_sorted]   # (V_kept, 3)
    kept_faces: List[List[int]] = [
        [int(old_to_new_orig[vi]) for vi in f]
        for f in unsel_faces
    ]

    n_kept = len(kept_verts)

    # --- 5. Map snapped boundary remeshed vertices to kept original verts ----
    # If a remeshed boundary vertex was snapped to an original boundary vertex
    # that is ALSO in the kept mesh, we should merge them (use the kept index
    # instead of appending a duplicate).
    final_remap = np.arange(len(snapped_verts), dtype=np.int32)  # rem_vi → final_vi

    for orig_vi, rem_vi in orig_bound_used.items():
        kept_new_idx = int(old_to_new_orig[orig_vi])
        if kept_new_idx >= 0:
            # This original boundary vertex IS in the kept mesh — reuse it
            final_remap[rem_vi] = kept_new_idx  # points into kept_verts range

    # Vertices not merged → appended after kept_verts
    append_positions = []
    rem_to_append_idx = np.full(len(snapped_verts), -1, dtype=np.int32)
    next_idx = n_kept

    for rem_vi in range(len(snapped_verts)):
        if final_remap[rem_vi] == rem_vi:
            # Not merged → append
            rem_to_append_idx[rem_vi] = next_idx
            append_positions.append(snapped_verts[rem_vi])
            next_idx += 1
        else:
            # Merged into a kept vertex — final_remap already correct
            pass

    # Fix up final_remap for non-merged vertices
    for rem_vi in range(len(snapped_verts)):
        if final_remap[rem_vi] == rem_vi and rem_to_append_idx[rem_vi] >= 0:
            final_remap[rem_vi] = rem_to_append_idx[rem_vi]

    # --- 6. Assemble final vertex array --------------------------------------
    if append_positions:
        new_verts = np.vstack([kept_verts, np.array(append_positions, dtype=np.float64)])
    else:
        new_verts = kept_verts

    # --- 7. Assemble final face lists ----------------------------------------
    new_faces: List[List[int]] = list(kept_faces)
    for f in remeshed_faces:
        new_faces.append([int(final_remap[vi]) for vi in f])

    return StitchResult(
        vertices=new_verts,
        faces=new_faces,
        boundary_snapped=snapped_count,
        boundary_total=len(rem_boundary),
        snap_distance_used=boundary_snap_distance,
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _find_boundary_vertices_from_faces(
    n_verts: int,
    faces: List[List[int]],
) -> Set[int]:
    """Return the set of vertex indices that sit on free (boundary) edges.

    A free edge is an edge referenced by exactly one face.
    """
    from collections import defaultdict
    edge_count: dict = defaultdict(int)

    for f in faces:
        for i in range(len(f)):
            v0 = f[i]
            v1 = f[(i + 1) % len(f)]
            key = (min(v0, v1), max(v0, v1))
            edge_count[key] += 1

    boundary_verts: Set[int] = set()
    for (v0, v1), count in edge_count.items():
        if count == 1:
            boundary_verts.add(v0)
            boundary_verts.add(v1)

    return boundary_verts


def build_selection_mask_from_face_indices(
    n_faces: int,
    selected_face_indices,
) -> np.ndarray:
    """Convenience: build a bool mask from a list/set of face indices."""
    mask = np.zeros(n_faces, dtype=bool)
    mask[list(selected_face_indices)] = True
    return mask
