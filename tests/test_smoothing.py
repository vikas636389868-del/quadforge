import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "engine" / "smoothing.py"
SPEC = importlib.util.spec_from_file_location("quadforge_smoothing", MODULE_PATH)
assert SPEC and SPEC.loader
smoothing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoothing)


def test_laplacian_step_with_csr_adjacency_does_not_crash():
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = [[0, 1, 2]]

    adjacency = smoothing.build_adjacency(faces, len(verts))
    csr = smoothing._build_csr_adjacency(adjacency)

    new_verts = smoothing._laplacian_step(verts, adjacency, 0.5, _csr=csr)

    assert new_verts.shape == verts.shape
    assert np.isfinite(new_verts).all()
