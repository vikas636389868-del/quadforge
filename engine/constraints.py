import numpy as np


def build_constraints(vertices: np.ndarray, faces: list, feature_edges: set) -> np.ndarray:
    """Build vertex constraint weights based on feature edges."""
    num_verts = len(vertices)
    if not feature_edges:
        return np.zeros(num_verts, dtype=np.float32)

    edges = np.array(list(feature_edges), dtype=np.int32)
    if edges.size == 0:
        return np.zeros(num_verts, dtype=np.float32)

    verts_idx = edges.flatten()
    verts_idx = verts_idx[(verts_idx >= 0) & (verts_idx < num_verts)]
    unique_verts = np.unique(verts_idx)

    weights = np.zeros(num_verts, dtype=np.float32)
    weights[unique_verts] = 1.0
    return weights
