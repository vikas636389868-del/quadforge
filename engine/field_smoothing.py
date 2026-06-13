"""Field smoothing compatibility wrapper.

The cross-field solver in field.py now includes global optimisation that
makes separate field smoothing less necessary. This module provides the
old API for backward compatibility.
"""

from .field import smooth_field, compute_face_directions  # noqa: F401

# Re-export build_face_adjacency for any code that imports it
def build_face_adjacency(faces, feature_edges=None):
    """Build face adjacency — legacy wrapper."""
    feature_edges = feature_edges or set()
    edge_map = {}
    adjacency = [set() for _ in range(len(faces))]

    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            v0 = face[i]
            v1 = face[(i + 1) % n]
            key = (v0, v1) if v0 < v1 else (v1, v0)
            if key in feature_edges:
                continue
            edge_map.setdefault(key, []).append(fi)

    for face_list in edge_map.values():
        if len(face_list) == 2:
            fi, fj = face_list
            adjacency[fi].add(fj)
            adjacency[fj].add(fi)

    return adjacency
