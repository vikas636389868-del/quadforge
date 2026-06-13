# parallel-hashmap 1.3.12

This directory contains the real **parallel-hashmap** header-only library
(version 1.3.12) by Gregory Popovitch, licensed under Apache-2.0.

Upstream: https://github.com/greg7mdp/parallel-hashmap
License:  see `LICENSE` in this directory.

## Layout

    parallel_hashmap/
        LICENSE                         Apache-2.0 licence text
        README.md                       (this file)
        parallel_hashmap/
            btree.h                     ordered B-tree containers
            meminfo.h                   process memory-usage helpers
            phmap.h                     main flat/node/parallel hash maps & sets
            phmap_base.h                base templates (Swiss-table core)
            phmap_bits.h                SIMD / bit-manipulation primitives
            phmap_config.h              compiler / platform feature macros
            phmap_dump.h                BinaryInputArchive / BinaryOutputArchive
            phmap_fwd_decl.h            forward declarations
            phmap_utils.h               HashState, hash_combine, Hash<T>

## Usage in QuadForge

Include in engine C++ sources with the canonical path:

```cpp
#include "parallel_hashmap/phmap.h"      // flat_hash_map etc.
#include "parallel_hashmap/btree.h"      // btree_map etc.
#include "parallel_hashmap/phmap_dump.h" // binary (de)serialization
```

The top-level CMake target `libquadforge` already adds
`QuadForge/engine/third_party/parallel_hashmap` to its include path, so no
extra configuration is required.

## Re-provisioning

If you need to refresh this directory (for example, to bump the version),
run:

```bash
python build/scripts/setup_third_party.py --force
```

The script downloads the pinned release tarball, verifies its SHA-256, and
copies the `parallel_hashmap/` subdirectory into this location.
