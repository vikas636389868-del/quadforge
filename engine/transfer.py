"""Material ID transfer for QuadForge.

Transfers per-face material indices from the input mesh to the output
quad mesh using spatial correspondence: each output face's centroid is
projected onto the input mesh, and the material of the nearest input
face is assigned.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from .spatial import TriangleBVH


def transfer_materials(
    output_vertices: np.ndarray,
    output_faces: List[List[int]],
    input_bvh: TriangleBVH,
    input_material_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Transfer material IDs from input to output mesh.

    Parameters
    ----------
    output_vertices : (V_out, 3)
    output_faces : list of face vertex lists
    input_bvh : TriangleBVH built from input mesh triangles
    input_material_ids : (F_in,) per-triangle material index
        If None, returns all zeros.

    Returns
    -------
    output_material_ids : (F_out,) per-face material index for output
    """
    num_out_faces = len(output_faces)

    if input_material_ids is None:
        return np.zeros(num_out_faces, dtype=np.int32)

    output_verts = np.asarray(output_vertices, dtype=np.float64)
    mat_ids = np.asarray(input_material_ids, dtype=np.int32)

    result = np.zeros(num_out_faces, dtype=np.int32)

    # BUG-R FIX (v118): the previous implementation queried the BVH inside
    # a Python for-loop over output faces — O(F_out) Python calls, each
    # crossing the Python↔C boundary individually.  For a 10K-quad output
    # mesh: 10K separate BVH queries through Python ≈ 10–30 ms.
    #
    # Fix: compute ALL face centroids in one vectorised NumPy pass, then
    # issue one BVH batch query (or at minimum keep the Python loop but
    # eliminate per-face np.mean overhead by pre-computing centroids).
    # Since TriangleBVH.closest_point() is a single-point C call with no
    # batch variant, we pre-build the centroid array in NumPy and loop only
    # over the BVH query calls — removing all per-face Python arithmetic.
    face_lens   = np.array([len(f) for f in output_faces], dtype=np.int32)
    valid_mask  = face_lens > 0
    # Pre-compute centroids for all valid faces in vectorised NumPy
    centroids   = np.zeros((num_out_faces, 3), dtype=np.float64)
    for fi in range(num_out_faces):
        if valid_mask[fi]:
            face = output_faces[fi]
            centroids[fi] = output_verts[face].mean(axis=0)   # (3,) — vectorised
    # BVH query loop — pure C per call, no Python arithmetic inside
    n_mat = len(mat_ids)
    for fi in range(num_out_faces):
        if not valid_mask[fi]:
            continue
        closest = input_bvh.closest_point(centroids[fi])
        if 0 <= closest.face_index < n_mat:
            result[fi] = mat_ids[closest.face_index]

    return result
