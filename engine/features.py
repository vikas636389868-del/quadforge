"""Feature edge detection for QuadForge.

Detects hard edges (dihedral angle), UV seams, material boundaries,
and custom split normal discontinuities using bmesh.
"""

import bpy
import bmesh
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class QFFeatures:
    hard_edges: set
    seam_edges: set
    material_edges: set
    normal_split_edges: set
    all_features: set


def _edge_key(v1, v2):
    return (v1, v2) if v1 < v2 else (v2, v1)


def detect_hard_edges(bm: bmesh.types.BMesh, angle_threshold_deg: float, include_boundary=True) -> set:
    angle_threshold = math.radians(angle_threshold_deg)
    hard_edges = set()

    for edge in bm.edges:
        if len(edge.link_faces) < 2:
            if include_boundary:
                v1, v2 = edge.verts[0].index, edge.verts[1].index
                hard_edges.add(_edge_key(v1, v2))
            continue

        f1, f2 = edge.link_faces[0], edge.link_faces[1]
        angle = f1.normal.angle(f2.normal)
        if angle > angle_threshold:
            v1, v2 = edge.verts[0].index, edge.verts[1].index
            hard_edges.add(_edge_key(v1, v2))

    return hard_edges


def detect_uv_seams(bm: bmesh.types.BMesh) -> set:
    seam_edges = set()
    for edge in bm.edges:
        if edge.seam:
            v1, v2 = edge.verts[0].index, edge.verts[1].index
            seam_edges.add(_edge_key(v1, v2))
    return seam_edges


def detect_material_edges(bm: bmesh.types.BMesh) -> set:
    material_edges = set()
    for edge in bm.edges:
        if len(edge.link_faces) < 2:
            continue
        f1, f2 = edge.link_faces[0], edge.link_faces[1]
        if f1.material_index != f2.material_index:
            v1, v2 = edge.verts[0].index, edge.verts[1].index
            material_edges.add(_edge_key(v1, v2))
    return material_edges


def detect_normal_split_edges(mesh_data, normal_angle_threshold_deg: float = 10.0) -> set:
    """Detect edges where custom split normals are discontinuous.

    When a mesh has custom split normals (e.g. from auto-smooth or
    manual normal editing), edges where the loop normals differ
    significantly indicate sharp features.

    Parameters
    ----------
    mesh_data : bpy.types.Mesh
    normal_angle_threshold_deg : float
        Angle threshold for normal discontinuity detection.

    Returns
    -------
    split_edges : set of (v0, v1) edge keys
    """
    split_edges = set()
    threshold = math.radians(normal_angle_threshold_deg)

    # Check if mesh has custom normals
    has_custom = False
    try:
        if hasattr(mesh_data, 'has_custom_normals'):
            has_custom = mesh_data.has_custom_normals
        elif hasattr(mesh_data, 'use_auto_smooth'):
            has_custom = mesh_data.use_auto_smooth
    except Exception:
        pass

    if not has_custom:
        return split_edges

    try:
        # Ensure loop normals are calculated
        mesh_data.calc_normals_split()

        # Build edge → loop mapping
        # Each edge has 2+ loops (one per adjacent face)
        # Compare normals of loops sharing the same vertex across the edge
        num_loops = len(mesh_data.loops)
        loop_normals = np.empty(num_loops * 3, dtype=np.float32)
        mesh_data.loops.foreach_get("normal", loop_normals)
        loop_normals = loop_normals.reshape((-1, 3))

        # For each edge, collect loop normals per vertex
        for edge in mesh_data.edges:
            if len(edge.vertices) != 2:
                continue
            v0, v1 = edge.vertices[0], edge.vertices[1]

            # Find all loops on this edge (loops whose vertex is v0 or v1,
            # belonging to faces that contain this edge)
            v0_normals = []
            v1_normals = []

            for poly in mesh_data.polygons:
                poly_verts = list(poly.vertices)
                if v0 not in poly_verts or v1 not in poly_verts:
                    continue
                # This face uses this edge
                for li in range(poly.loop_start, poly.loop_start + poly.loop_total):
                    loop = mesh_data.loops[li]
                    if loop.vertex_index == v0:
                        v0_normals.append(loop_normals[li])
                    elif loop.vertex_index == v1:
                        v1_normals.append(loop_normals[li])

            # Check if normals differ significantly at either vertex
            is_split = False
            for normals_list in [v0_normals, v1_normals]:
                if len(normals_list) >= 2:
                    for i in range(len(normals_list)):
                        for j in range(i + 1, len(normals_list)):
                            n1 = normals_list[i]
                            n2 = normals_list[j]
                            dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
                            angle = math.acos(dot)
                            if angle > threshold:
                                is_split = True
                                break
                        if is_split:
                            break
                if is_split:
                    break

            if is_split:
                split_edges.add(_edge_key(v0, v1))

        mesh_data.free_normals_split()
    except Exception:
        pass

    return split_edges


def extract_features(obj: bpy.types.Object, settings) -> QFFeatures:
    """Extract all feature edges from a Blender mesh object.

    Parameters
    ----------
    obj : bpy.types.Object
    settings : QFSettingsPropertyGroup or _SettingsSnapshot
    """
    mesh = obj.data
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bm.verts.index_update()
        bm.faces.index_update()
        bm.edges.index_update()
        bm.normal_update()

        hard_edges = set()
        seam_edges = set()
        material_edges = set()
        normal_split_edges = set()

        if getattr(settings, 'auto_detect_hard_edges', True):
            angle = getattr(settings, 'hard_edge_angle_deg', 30.0)
            hard_edges = detect_hard_edges(bm, angle, include_boundary=True)

        if getattr(settings, 'use_uv_seams', False):
            seam_edges = detect_uv_seams(bm)

        if getattr(settings, 'use_materials', False):
            material_edges = detect_material_edges(bm)

        if getattr(settings, 'use_normals', False):
            normal_split_edges = detect_normal_split_edges(mesh)

        all_features = hard_edges | seam_edges | material_edges | normal_split_edges

        return QFFeatures(
            hard_edges=hard_edges,
            seam_edges=seam_edges,
            material_edges=material_edges,
            normal_split_edges=normal_split_edges,
            all_features=all_features,
        )
    finally:
        bm.free()


# ---------------------------------------------------------------------------
# Non-bmesh feature detection (works from HalfEdgeMesh, no bpy needed)
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass


@_dataclass
class FeatureData:
    """Feature detection result for tests and non-Blender usage."""
    feature_edges: set
    hard_edges: set
    seam_edges: set     = None
    material_edges: set = None
    normal_split_edges: set = None

    @property
    def all_features(self):
        return self.feature_edges


def detect_features(mesh_or_he, angle_threshold_deg: float = 30.0) -> FeatureData:
    """Detect feature edges on a HalfEdgeMesh or (vertices, faces) pair.

    Compatible with both the Blender-based extract_features() and the
    pure-Python test environment.

    Parameters
    ----------
    mesh_or_he : HalfEdgeMesh or object with .vertices / .faces attributes
    angle_threshold_deg : float, dihedral angle threshold in degrees

    Returns
    -------
    FeatureData with .feature_edges (a set of (v0, v1) tuples).
    """
    import math
    import numpy as np

    # Extract vertices and faces
    if hasattr(mesh_or_he, 'vertices') and (hasattr(mesh_or_he, 'faces') or hasattr(mesh_or_he, '_face_lists')):
        he = mesh_or_he
        vertices = np.asarray(he.vertices, dtype=np.float64)
        if hasattr(he, 'faces'):
            faces = list(he.faces)
        else:
            faces = list(he._face_lists)
        faces_arr = np.asarray(faces, dtype=np.int32)
        edge_to_faces = getattr(he, 'edge_to_faces', None)
    else:
        raise TypeError(f"detect_features: unsupported input type {type(mesh_or_he)}")

    threshold_rad = math.radians(angle_threshold_deg)
    hard_edges: set = set()

    # Compute face normals
    v0 = vertices[faces_arr[:, 0]]
    v1 = vertices[faces_arr[:, 1]]
    v2 = vertices[faces_arr[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    lens = np.linalg.norm(cross, axis=1, keepdims=True)
    lens[lens < 1e-15] = 1.0
    face_normals = cross / lens  # (F, 3)

    # Build edge → face adjacency
    if edge_to_faces is not None:
        e2f = edge_to_faces
    else:
        e2f: dict = {}
        for fi, face in enumerate(faces):
            n = len(face)
            for i in range(n):
                va = face[i]; vb = face[(i + 1) % n]
                key = (va, vb) if va < vb else (vb, va)
                e2f.setdefault(key, []).append(fi)

    for edge_key, adj in e2f.items():
        if len(adj) < 2:
            # Boundary edge — always a feature
            hard_edges.add(edge_key)
            continue
        fi, fj = adj[0], adj[1]
        dot = float(np.clip(np.dot(face_normals[fi], face_normals[fj]), -1.0, 1.0))
        angle = math.acos(dot)
        if angle > threshold_rad:
            hard_edges.add(edge_key)

    return FeatureData(
        feature_edges=hard_edges,
        hard_edges=hard_edges,
        seam_edges=set(),
        material_edges=set(),
        normal_split_edges=set(),
    )
