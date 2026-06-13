"""Output enhancement for QuadForge.

Post-processing enhancements applied to the output mesh before
returning it to Blender:

  - UV Transfer: Projects UVs from the input mesh to the output mesh
    using barycentric interpolation from the BVH closest-point query
  - Shade Transfer: Copies flat/smooth shading from input to output
  - Normal Transfer: Optionally transfers custom normals
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from .spatial import TriangleBVH


def transfer_uvs(
    output_vertices: np.ndarray,
    output_faces: List[List[int]],
    input_vertices: np.ndarray,
    input_faces: np.ndarray,
    input_uv_coords: np.ndarray,
    input_bvh: TriangleBVH,
) -> Optional[np.ndarray]:
    """Transfer UV coordinates from input to output mesh.

    For each output vertex, find the closest point on the input mesh
    and interpolate the UV using barycentric coordinates.

    Parameters
    ----------
    output_vertices : (V_out, 3) output mesh vertex positions
    output_faces : list of face vertex lists
    input_vertices : (V_in, 3) input mesh vertices
    input_faces : (F_in, 3) input triangle faces
    input_uv_coords : (L_in, 2) per-loop UV coordinates
    input_bvh : BVH built from input mesh

    Returns
    -------
    output_uvs : (V_out, 2) per-vertex UV coordinates, or None if failed
    """
    if input_uv_coords is None or len(input_uv_coords) == 0:
        return None

    output_verts = np.asarray(output_vertices, dtype=np.float64)
    input_verts = np.asarray(input_vertices, dtype=np.float64)
    input_faces_arr = np.asarray(input_faces, dtype=np.int32)
    num_out_verts = len(output_verts)

    # Build per-triangle UV lookup
    # Each triangle has 3 loop UVs
    # input_uv_coords is indexed by loop, and each triangle fi has
    # loops at fi*3, fi*3+1, fi*3+2 (for calc_loop_triangles output)
    num_input_tris = len(input_faces_arr)

    output_uvs = np.zeros((num_out_verts, 2), dtype=np.float32)

    for vi in range(num_out_verts):
        closest = input_bvh.closest_point(output_verts[vi])
        fi = closest.face_index
        bary = closest.bary

        if fi < 0 or fi >= num_input_tris:
            continue

        # Get loop UVs for this triangle
        # The UVs are stored per-loop; for triangulated mesh from
        # calc_loop_triangles, triangle fi's loops are at index fi*3
        loop_base = fi * 3
        if loop_base + 2 < len(input_uv_coords):
            uv0 = input_uv_coords[loop_base]
            uv1 = input_uv_coords[loop_base + 1]
            uv2 = input_uv_coords[loop_base + 2]
            output_uvs[vi] = bary[0] * uv0 + bary[1] * uv1 + bary[2] * uv2
        else:
            # Fallback: use vertex UVs if loop mapping fails
            tri = input_faces_arr[fi]
            # Try per-vertex UV (some meshes have these)
            for k in range(3):
                vert_idx = tri[k]
                if vert_idx < len(input_uv_coords):
                    output_uvs[vi] += bary[k] * input_uv_coords[vert_idx]

    return output_uvs


def transfer_vertex_colors(
    output_vertices: np.ndarray,
    input_vertices: np.ndarray,
    input_vertex_colors: np.ndarray,
    input_faces: np.ndarray,
    input_bvh: TriangleBVH,
) -> Optional[np.ndarray]:
    """Transfer vertex colors from input to output mesh.

    Parameters
    ----------
    output_vertices : (V_out, 3)
    input_vertices : (V_in, 3)
    input_vertex_colors : (V_in, 3) or (V_in, 4) RGB[A] colors
    input_faces : (F_in, 3)
    input_bvh : TriangleBVH

    Returns
    -------
    output_colors : (V_out, 3) RGB colors per output vertex, or None
    """
    if input_vertex_colors is None or len(input_vertex_colors) == 0:
        return None

    colors = np.asarray(input_vertex_colors, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[1] < 3:
        return None

    input_faces_arr = np.asarray(input_faces, dtype=np.int32)
    output_verts = np.asarray(output_vertices, dtype=np.float64)
    num_out = len(output_verts)

    output_colors = np.zeros((num_out, 3), dtype=np.float32)

    for vi in range(num_out):
        closest = input_bvh.closest_point(output_verts[vi])
        fi = closest.face_index
        bary = closest.bary

        if fi < 0 or fi >= len(input_faces_arr):
            continue

        tri = input_faces_arr[fi]
        for k in range(3):
            vert_idx = tri[k]
            if vert_idx < len(colors):
                output_colors[vi] += bary[k] * colors[vert_idx, :3]

    return output_colors


def compute_output_face_shading(
    output_vertices: np.ndarray,
    output_faces: List[List[int]],
    input_bvh: TriangleBVH,
    input_face_smooth: np.ndarray,
) -> np.ndarray:
    """Determine per-face smooth/flat shading for output mesh.

    Each output face inherits the shading of its nearest input face.

    Parameters
    ----------
    output_vertices : (V_out, 3)
    output_faces : list of face vertex lists
    input_bvh : TriangleBVH from input
    input_face_smooth : (F_in,) bool array — True for smooth shaded

    Returns
    -------
    output_smooth : (F_out,) bool array
    """
    if input_face_smooth is None:
        return np.ones(len(output_faces), dtype=bool)

    output_verts = np.asarray(output_vertices, dtype=np.float64)
    output_smooth = np.ones(len(output_faces), dtype=bool)

    for fi, face in enumerate(output_faces):
        if not face:
            continue
        centroid = np.mean(output_verts[face], axis=0)
        closest = input_bvh.closest_point(centroid)
        if 0 <= closest.face_index < len(input_face_smooth):
            output_smooth[fi] = input_face_smooth[closest.face_index]

    return output_smooth


def compute_output_normals(
    output_vertices: np.ndarray,
    output_faces: List[List[int]],
) -> np.ndarray:
    """Compute proper per-vertex normals for the output quad mesh.

    Uses area-weighted face normal averaging, which works correctly
    for both quad and triangle faces.

    Parameters
    ----------
    output_vertices : (V, 3)
    output_faces : list of face vertex lists

    Returns
    -------
    normals : (V, 3) unit per-vertex normals
    """
    verts = np.asarray(output_vertices, dtype=np.float64)
    num_verts = len(verts)
    normals = np.zeros((num_verts, 3), dtype=np.float64)

    for face in output_faces:
        n = len(face)
        if n < 3:
            continue

        # Compute face normal using Newell's method
        face_normal = np.zeros(3, dtype=np.float64)
        for i in range(n):
            v0 = verts[face[i]]
            v1 = verts[face[(i + 1) % n]]
            face_normal[0] += (v0[1] - v1[1]) * (v0[2] + v1[2])
            face_normal[1] += (v0[2] - v1[2]) * (v0[0] + v1[0])
            face_normal[2] += (v0[0] - v1[0]) * (v0[1] + v1[1])

        # Area weight is implicit (cross product magnitude ∝ area)
        for vi in face:
            normals[vi] += face_normal

    # Normalise
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-15] = 1.0
    normals /= lengths

    return normals.astype(np.float32)
