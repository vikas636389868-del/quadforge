import importlib.util
import pathlib
import numpy as np

path = pathlib.Path('engine/smoothing.py')
spec = importlib.util.spec_from_file_location('quadforge_smoothing', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
verts = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float64)
faces = [[0, 1, 2]]
adj = module.build_adjacency(faces, len(verts))
csr = module._build_csr_adjacency(adj)
module._laplacian_step(verts, adj, 0.5, _csr=csr)
print('ok')
