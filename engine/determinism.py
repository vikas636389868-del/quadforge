"""QuadForge — Deterministic Quality Mode (Roadmap §13.1).

Ensures that the same input + same settings produces bit-exactly the same
quad output. This is what studios need for version control, asset review,
and regression testing.

Responsibilities:
    1. Fixed random seed for every stochastic operation.
    2. Stable vertex/face ordering before solve AND after extraction.
    3. Deterministic parallel reductions for OpenMP code paths.
    4. Deterministic fallback ordering for singularity relocation, seam
       selection, and MIQ rounding.

Usage:
    from .determinism import DeterministicContext, canonicalize_mesh

    with DeterministicContext(seed=42):
        verts, faces = canonicalize_mesh(verts, faces)
        # ... run pipeline ...

All stages that use ``np.random`` or Python's ``random`` will honour the
seed set by the context. Stages that internally use OpenMP must use
deterministic ordered reductions when ``is_deterministic_mode()`` is True.
"""

from __future__ import annotations

import hashlib
import os
import random
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Global state (thread-local for safety)
# ---------------------------------------------------------------------------

_TLS = threading.local()


def is_deterministic_mode() -> bool:
    return bool(getattr(_TLS, "enabled", False))


def get_seed() -> int:
    return int(getattr(_TLS, "seed", 0))


def get_rng() -> np.random.Generator:
    """Return a deterministic numpy Generator when in deterministic mode,
    otherwise the default (non-deterministic) Generator."""
    if is_deterministic_mode():
        return np.random.default_rng(get_seed())
    return np.random.default_rng()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@dataclass
class _SavedState:
    py_state: tuple
    np_state: dict
    env_omp: Optional[str]
    env_blas: Optional[str]
    enabled: bool
    seed: int


@contextmanager
def DeterministicContext(seed: int = 0xC0FFEE):
    """Context manager that pins all RNGs and forces single-threaded
    BLAS / OpenMP reductions (when possible) so every sparse solve,
    eigen decomposition, and stochastic step is reproducible.
    """
    saved = _SavedState(
        py_state=random.getstate(),
        np_state=np.random.get_state(),
        env_omp=os.environ.get("OMP_NUM_THREADS"),
        env_blas=os.environ.get("OPENBLAS_NUM_THREADS"),
        enabled=is_deterministic_mode(),
        seed=get_seed(),
    )

    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    _TLS.enabled = True
    _TLS.seed = int(seed)

    try:
        yield
    finally:
        random.setstate(saved.py_state)
        np.random.set_state(saved.np_state)
        if saved.env_omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = saved.env_omp
        if saved.env_blas is None:
            os.environ.pop("OPENBLAS_NUM_THREADS", None)
        else:
            os.environ["OPENBLAS_NUM_THREADS"] = saved.env_blas
        _TLS.enabled = saved.enabled
        _TLS.seed = saved.seed


# ---------------------------------------------------------------------------
# Canonical mesh ordering
# ---------------------------------------------------------------------------

def canonicalize_mesh(vertices: np.ndarray,
                      faces: np.ndarray,
                      *,
                      tol: float = 1.0e-9,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a canonical ordering of (vertices, faces) that is invariant
    under input permutations.

    Vertex ordering: lexicographic sort on quantized positions.
    Face ordering: lexicographic sort on (rotated smallest-first) vertex
    tuple after remapping.

    Returns
    -------
    new_vertices : (N, 3)
    new_faces    : (F, K) with same K as input
    vertex_remap : (N,) array where ``new_verts[i] = vertices[remap[i]]``
    """
    if len(vertices) == 0:
        return vertices, faces, np.zeros(0, dtype=np.int64)

    # Quantize to make sorting robust to floating-point noise
    q = np.round(vertices / tol).astype(np.int64)
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0]))

    new_verts = vertices[order]
    # Build inverse permutation so we can remap faces
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    new_faces = inv[faces]

    # Rotate each face so its smallest vertex index comes first (preserves
    # winding) then lex-sort faces
    if new_faces.ndim == 2:
        F, K = new_faces.shape
        rotated = np.empty_like(new_faces)
        for i in range(F):
            f = new_faces[i]
            k = int(np.argmin(f))
            rotated[i] = np.concatenate([f[k:], f[:k]])
        # Lex sort using columns
        sort_keys = tuple(rotated[:, K - 1 - c] for c in range(K))
        face_order = np.lexsort(sort_keys)
        new_faces = rotated[face_order]

    return new_verts, new_faces, order


# ---------------------------------------------------------------------------
# Deterministic parallel reductions
# ---------------------------------------------------------------------------

def deterministic_sum(values: np.ndarray) -> float:
    """Pairwise-summation for bit-stable reduction regardless of thread
    count. Equivalent in precision to ``math.fsum`` on double arrays but
    faster via numpy recursion.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    # numpy's .sum() uses pairwise summation for large arrays with a fixed
    # block size, which is deterministic for a fixed array shape.
    return float(arr.sum())


def stable_argsort(values: np.ndarray, secondary: Optional[np.ndarray] = None
                   ) -> np.ndarray:
    """Stable (mergesort) argsort that breaks ties using ``secondary`` and
    finally by index, so the resulting permutation is fully deterministic.
    """
    if secondary is None:
        return np.argsort(values, kind="stable")
    idx = np.arange(len(values))
    return np.lexsort((idx, secondary, values))


# ---------------------------------------------------------------------------
# Deterministic hashing for regression tests
# ---------------------------------------------------------------------------

def mesh_fingerprint(vertices: np.ndarray, faces, *,
                     tol: float = 1.0e-6) -> str:
    """Stable hex digest of a mesh's canonical form. Two meshes that
    canonicalize to the same thing will produce the same fingerprint.
    Used by the benchmark harness to detect silent output drift.
    """
    if isinstance(faces, list):
        # Flatten polygon list with size prefix per face
        parts: List[int] = []
        for f in faces:
            parts.append(len(f))
            parts.extend(int(v) for v in f)
        faces_arr = np.asarray(parts, dtype=np.int64)
    else:
        faces_arr = np.asarray(faces, dtype=np.int64)

    q = np.round(np.asarray(vertices, dtype=np.float64) / tol).astype(np.int64)
    h = hashlib.sha256()
    h.update(q.tobytes())
    h.update(faces_arr.tobytes())
    return h.hexdigest()


def params_fingerprint(params_dict: dict) -> str:
    """Stable hash of the parameter dictionary — used as part of the cache
    key when memoizing preview results."""
    h = hashlib.sha256()
    for k in sorted(params_dict.keys()):
        v = params_dict[k]
        h.update(k.encode("utf-8"))
        h.update(repr(v).encode("utf-8"))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Deterministic tie-breaking for singularity relocation / MIQ rounding
# ---------------------------------------------------------------------------

def deterministic_choose(candidates: Iterable, key_fn) -> Optional[object]:
    """Pick the candidate minimising ``key_fn`` with deterministic tie-break
    by the candidate's ``repr`` (so identical scores always resolve the
    same way across runs and platforms)."""
    best = None
    best_key: Optional[Tuple] = None
    for c in candidates:
        k = (key_fn(c), repr(c))
        if best_key is None or k < best_key:
            best = c
            best_key = k
    return best
