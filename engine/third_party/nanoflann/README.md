# nanoflann (vendored)

This directory contains the **real** nanoflann single-header KD-tree library,
vendored into the QuadForge source tree so the engine can be built without
any external package fetches.

- Header:   `nanoflann.hpp`   — single-header library (`NANOFLANN_VERSION 0x190`)
- Licence:  `COPYING`         — BSD-2-Clause, © 2008-2024 J. L. Blanco et al.
- Upstream: https://github.com/jlblancoc/nanoflann

## What it provides to QuadForge

`engine/src/mesh/spatial.cpp` uses nanoflann to build a KD-tree over triangle
centroids for the `TriangleBVH` spatial index. The public API used is:

- `nanoflann::KDTreeSingleIndexAdaptor<L2_Adaptor<double, Cloud>, Cloud, 3>`
- `KDTreeSingleIndexAdaptorParams(leaf_max_size)`
- `buildIndex()`
- `knnSearch(query, k, out_indices, out_dists)`
- `radiusSearch(query, radius2, result_items)`

This gives the O(log N) nearest-point and radius queries that the smoothing,
surface projection, feature snapping and material transfer passes in the
remeshing pipeline all rely on. The previous stub header shipped in this
directory performed an O(N) linear scan and has been fully replaced.

## Licence note

nanoflann is redistributed under the BSD-2-Clause licence shown in `COPYING`.
No modifications have been made to the upstream header.
