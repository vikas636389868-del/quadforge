"""QuadForge v1.2 — GPU Acceleration Python Interface.

Provides the Python-side GPU solver detection, selection, and fallback
logic for the sparse conjugate gradient solver used in the cross-field
and parametrization stages.

Three backends are probed at startup (in priority order):
  1. CUDA CG via the compiled libquadforge native library  (NVIDIA GPUs)
  2. Metal Compute via the compiled libquadforge native library (Apple Silicon)
  3. PyOpenCL (if installed) — cross-platform fallback
  4. CPU (always available) — OpenMP-accelerated Eigen solver

Only one backend is active per session.  The active backend is
selected at add-on load time (or when the user presses "Detect GPU")
and cached for the lifetime of the add-on.

Public API
----------
    detect_gpu() -> GPUInfo
        Probe and return information about the best available GPU.

    get_active_backend() -> str
        Return the currently active solver backend name.

    set_backend(name: str) -> bool
        Explicitly select a backend.  Returns True on success.

    GPUInfo
        Dataclass: available, backend, device_name, vram_mb, driver_version

    SparseGPUSolver
        Thin wrapper: wraps either the native CG call or a scipy fallback.
        Used internally by field.py and parametrize.py when use_gpu=True.
"""

from __future__ import annotations

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("QuadForge.gpu_solver")


# ---------------------------------------------------------------------------
# GPUInfo dataclass
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Information about the detected GPU and solver backend.

    Attributes
    ----------
    available : bool
        True if any GPU backend is available.
    backend : str
        One of: 'cuda', 'metal', 'opencl', 'cpu'.
    device_name : str
        Human-readable device name (e.g. 'NVIDIA RTX 3080').
    vram_mb : int
        Estimated VRAM in megabytes.  0 if unknown.
    driver_version : str
        Driver or runtime version string.  '' if unknown.
    compute_capability : str
        CUDA compute capability string (e.g. '8.6').  '' if not CUDA.
    supports_double : bool
        True if the device supports double-precision float64.
    estimated_speedup : float
        Estimated speedup over CPU for the sparse CG solve (heuristic).
    notes : list[str]
        Human-readable notes / warnings collected during probing.
    """
    available: bool = False
    backend: str = "cpu"
    device_name: str = "CPU (no GPU detected)"
    vram_mb: int = 0
    driver_version: str = ""
    compute_capability: str = ""
    supports_double: bool = True
    estimated_speedup: float = 1.0
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        """One-line summary for the UI."""
        if not self.available:
            return f"CPU solver (no GPU) — {self.device_name}"
        return (
            f"{self.backend.upper()} — {self.device_name} "
            f"({self.vram_mb} MB VRAM, ~{self.estimated_speedup:.1f}× speedup)"
        )


# ---------------------------------------------------------------------------
# Backend probing
# ---------------------------------------------------------------------------

_gpu_info_cache: Optional[GPUInfo] = None
_active_backend: str = "cpu"


def _probe_cuda() -> Optional[GPUInfo]:
    """Try to detect an NVIDIA CUDA-capable GPU."""
    try:
        # Try nvidia-smi first (most reliable, no CUDA toolkit needed)
        import subprocess
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None

        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return None

        # Parse first GPU
        parts = [p.strip() for p in lines[0].split(",")]
        name = parts[0] if len(parts) > 0 else "NVIDIA GPU"
        vram = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        driver = parts[2] if len(parts) > 2 else ""
        cc = parts[3] if len(parts) > 3 else ""

        # Estimate speedup heuristic based on VRAM
        speedup = 1.0
        if vram >= 8000:
            speedup = 4.0
        elif vram >= 4000:
            speedup = 2.5
        elif vram >= 2000:
            speedup = 1.5

        return GPUInfo(
            available=True,
            backend="cuda",
            device_name=name,
            vram_mb=vram,
            driver_version=driver,
            compute_capability=cc,
            supports_double=True,
            estimated_speedup=speedup,
            notes=[f"{len(lines)} GPU(s) found; using first"],
        )

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # Fallback: try pycuda
    try:
        import pycuda.driver as drv  # type: ignore
        drv.init()
        if drv.Device.count() == 0:
            return None
        dev = drv.Device(0)
        vram = dev.total_memory() // (1024 * 1024)
        return GPUInfo(
            available=True,
            backend="cuda",
            device_name=dev.name(),
            vram_mb=vram,
            driver_version=str(drv.get_version()),
            compute_capability=".".join(str(x) for x in dev.compute_capability()),
            supports_double=True,
            estimated_speedup=max(1.5, vram / 2000),
            notes=["Detected via pycuda"],
        )
    except ImportError:
        pass
    except Exception as exc:
        logger.debug(f"CUDA probe failed: {exc}")

    return None


def _probe_metal() -> Optional[GPUInfo]:
    """Try to detect Apple Metal (macOS only)."""
    if sys.platform != "darwin":
        return None
    try:
        import subprocess
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None

        text = result.stdout
        # Look for Apple Silicon GPU
        if "Apple M" in text or "Metal" in text:
            # Extract GPU name
            gpu_name = "Apple GPU"
            for line in text.splitlines():
                if "Chipset Model" in line or "Metal:" in line:
                    gpu_name = line.split(":")[-1].strip()
                    break

            # Apple Silicon unified memory — estimate from system RAM
            try:
                mem_result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=2
                )
                ram_bytes = int(mem_result.stdout.strip())
                # Metal uses shared memory; allow ~40% for GPU
                vram_mb = int(ram_bytes / 1024 / 1024 * 0.4)
            except Exception:
                vram_mb = 8192

            return GPUInfo(
                available=True,
                backend="metal",
                device_name=gpu_name,
                vram_mb=vram_mb,
                driver_version="Metal (system)",
                compute_capability="",
                supports_double=False,   # Metal < 3.0 has limited fp64
                estimated_speedup=2.5,
                notes=["Apple Metal — unified memory GPU"],
            )
    except Exception as exc:
        logger.debug(f"Metal probe failed: {exc}")

    return None


def _probe_opencl() -> Optional[GPUInfo]:
    """Try to detect any OpenCL GPU."""
    try:
        import pyopencl as cl  # type: ignore
        platforms = cl.get_platforms()
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if not devices:
                continue
            dev = devices[0]
            vram_mb = dev.global_mem_size // (1024 * 1024)
            fp64 = bool(dev.double_fp_config)
            return GPUInfo(
                available=True,
                backend="opencl",
                device_name=dev.name.strip(),
                vram_mb=vram_mb,
                driver_version=platform.version,
                compute_capability="",
                supports_double=fp64,
                estimated_speedup=max(1.2, vram_mb / 4000),
                notes=[f"OpenCL platform: {platform.name}"],
            )
    except ImportError:
        pass
    except Exception as exc:
        logger.debug(f"OpenCL probe failed: {exc}")

    return None


def _probe_native_gpu() -> Optional[GPUInfo]:
    """Check if the compiled libquadforge has GPU support compiled in.

    The native library exposes qf_has_gpu() which returns 1 if it was
    compiled with CUDA or Metal support.
    """
    try:
        from QuadForge.bridge import _lib  # may be None if no native lib
        if _lib is None:
            return None
        has_gpu = getattr(_lib, "qf_has_gpu", None)
        if has_gpu is None:
            return None
        if has_gpu() == 1:
            gpu_version = getattr(_lib, "qf_gpu_device_name", None)
            name = gpu_version().decode("utf-8") if gpu_version else "Native GPU"
            return GPUInfo(
                available=True,
                backend="cuda",
                device_name=name,
                vram_mb=0,
                driver_version="native",
                estimated_speedup=3.0,
                notes=["Detected via libquadforge native qf_has_gpu()"],
            )
    except Exception:
        pass
    return None


def detect_gpu() -> GPUInfo:
    """Probe and return information about the best available GPU solver.

    Results are cached — repeated calls are instant after the first.

    The probing order is:
      1. Native libquadforge GPU (if compiled with CUDA/Metal)
      2. CUDA (nvidia-smi or pycuda)
      3. Metal (macOS only)
      4. OpenCL (pyopencl)
      5. CPU fallback
    """
    global _gpu_info_cache, _active_backend

    if _gpu_info_cache is not None:
        return _gpu_info_cache

    # Try backends in priority order
    probers = [
        _probe_native_gpu,
        _probe_cuda,
        _probe_metal,
        _probe_opencl,
    ]

    for probe_fn in probers:
        try:
            info = probe_fn()
            if info is not None:
                _gpu_info_cache = info
                _active_backend = info.backend
                logger.info(f"[QuadForge] GPU backend: {info.summary()}")
                return info
        except Exception as exc:
            logger.debug(f"Probe {probe_fn.__name__} raised: {exc}")

    # CPU fallback
    _gpu_info_cache = GPUInfo(
        available=False,
        backend="cpu",
        device_name="CPU (OpenMP)",
        notes=["No GPU backend detected — using CPU solver"],
    )
    _active_backend = "cpu"
    logger.info("[QuadForge] No GPU — using CPU solver")
    return _gpu_info_cache


def get_active_backend() -> str:
    """Return the currently active solver backend name ('cuda', 'metal', 'opencl', 'cpu')."""
    return _active_backend


def set_backend(name: str) -> bool:
    """Explicitly select a backend.

    Parameters
    ----------
    name : str
        One of: 'cuda', 'metal', 'opencl', 'cpu', 'auto'.

    Returns True if the backend was set, False if unsupported.
    """
    global _active_backend, _gpu_info_cache

    if name == "auto":
        _gpu_info_cache = None   # force re-probe
        detect_gpu()
        return True

    if name == "cpu":
        _active_backend = "cpu"
        return True

    # Probe to verify the requested backend is actually available
    probers = {"cuda": _probe_cuda, "metal": _probe_metal, "opencl": _probe_opencl}
    probe_fn = probers.get(name)
    if probe_fn is None:
        logger.warning(f"Unknown backend '{name}'")
        return False

    info = probe_fn()
    if info is not None:
        _active_backend = name
        _gpu_info_cache = info
        return True

    logger.warning(f"Backend '{name}' not available on this system")
    return False


def reset_cache() -> None:
    """Force re-probe on next detect_gpu() call (used after driver updates)."""
    global _gpu_info_cache
    _gpu_info_cache = None


# ---------------------------------------------------------------------------
# SparseGPUSolver
# ---------------------------------------------------------------------------

class SparseGPUSolver:
    """Thin wrapper for sparse linear system Ax=b solvers.

    Selects the best available solver automatically, with fallback to
    scipy.sparse.linalg.cg when no GPU is available.

    This class is used internally by:
      - engine/field.py  (connection Laplacian eigensolver)
      - engine/parametrize.py  (Poisson / MIQ solve)

    Usage
    -----
    >>> solver = SparseGPUSolver(A_csr)
    >>> x = solver.solve(b)

    The solver caches the factorization (or preconditioner) on the first
    call so subsequent solves with different right-hand sides are fast.
    """

    def __init__(self, A, use_gpu: Optional[bool] = None, tol: float = 1e-6,
                 max_iter: int = 1000, verbose: bool = False):
        """
        Parameters
        ----------
        A : scipy.sparse.spmatrix  (CSR preferred)
            The sparse coefficient matrix.
        use_gpu : bool, optional
            True to force GPU, False to force CPU, None for auto-detect.
        tol : float
            Convergence tolerance for iterative solvers.
        max_iter : int
            Maximum iterations for the CG solver.
        verbose : bool
            Print solver progress.
        """
        self._A = A
        self._tol = tol
        self._max_iter = max_iter
        self._verbose = verbose
        self._backend = "cpu"
        self._precond = None
        self._factored = None

        # Determine backend
        if use_gpu is False:
            self._backend = "cpu"
        elif use_gpu is True or (use_gpu is None and _active_backend != "cpu"):
            info = detect_gpu()
            self._backend = info.backend if info.available else "cpu"
        else:
            self._backend = "cpu"

        # Pre-build a preconditioner (diagonal / ILU) for the CPU path
        self._build_preconditioner()

    def _build_preconditioner(self):
        """Build an incomplete Cholesky / diagonal preconditioner."""
        try:
            from scipy.sparse.linalg import spilu, LinearOperator
            import scipy.sparse as sp

            A = self._A
            if not sp.issparse(A):
                return

            # Try ILU — excellent for SPD systems
            try:
                ilu = spilu(A.tocsc(), fill_factor=4, drop_tol=1e-4)
                n = A.shape[0]
                self._precond = LinearOperator(
                    (n, n), matvec=ilu.solve, dtype=A.dtype
                )
            except Exception:
                # Fallback: diagonal (Jacobi) preconditioner
                diag = np.array(A.diagonal(), dtype=np.float64)
                diag[np.abs(diag) < 1e-15] = 1.0
                inv_diag = 1.0 / diag
                n = A.shape[0]
                self._precond = LinearOperator(
                    (n, n),
                    matvec=lambda x: inv_diag * x,
                    dtype=np.float64,
                )
        except ImportError:
            pass

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve Ax = b.

        Parameters
        ----------
        b : np.ndarray, shape (n,)  or (n, k) for multiple right-hand sides.

        Returns
        -------
        x : np.ndarray, same shape as b.
        """
        if b.ndim == 2:
            # Multiple right-hand sides: solve column by column
            return np.column_stack([self.solve(b[:, i]) for i in range(b.shape[1])])

        if self._backend in ("cuda", "metal") and self._A is not None:
            result = self._solve_gpu(b)
            if result is not None:
                return result
            # GPU failed — fall through to CPU

        return self._solve_cpu(b)

    def _solve_gpu(self, b: np.ndarray) -> Optional[np.ndarray]:
        """Attempt GPU-accelerated CG solve via native library."""
        try:
            from QuadForge.bridge import _lib
            if _lib is None:
                return None
            # If the native library exports a GPU solver, call it.
            # Otherwise return None to fall back.
            gpu_solve = getattr(_lib, "qf_gpu_sparse_cg", None)
            if gpu_solve is None:
                return None

            import ctypes
            n = len(b)
            x = np.zeros(n, dtype=np.float64)
            b_c = b.astype(np.float64)
            gpu_solve(
                ctypes.c_int(n),
                b_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_double(self._tol),
                ctypes.c_int(self._max_iter),
            )
            return x
        except Exception as exc:
            logger.debug(f"GPU solve failed: {exc}")
            return None

    def _solve_cpu(self, b: np.ndarray) -> np.ndarray:
        """CPU sparse CG solve using scipy."""
        try:
            from scipy.sparse.linalg import cg

            b64 = b.astype(np.float64)
            x0 = np.zeros_like(b64)

            x, info = cg(
                self._A, b64, x0=x0,
                tol=self._tol,
                maxiter=self._max_iter,
                M=self._precond,
            )

            if self._verbose and info != 0:
                logger.debug(f"CG did not converge: info={info}")

            return x

        except ImportError:
            # scipy not available: use numpy dense solve (last resort)
            try:
                import scipy.sparse as sp
                A_dense = self._A.toarray() if sp.issparse(self._A) else self._A
            except ImportError:
                A_dense = np.array(self._A.todense()) if hasattr(self._A, 'todense') else self._A
            return np.linalg.solve(A_dense.astype(np.float64), b.astype(np.float64))

    def solve_eigen(self, k: int = 1) -> tuple:
        """Find the k smallest eigenvectors of A (for the cross-field solver).

        Returns
        -------
        (eigenvalues, eigenvectors) : (np.ndarray (k,), np.ndarray (n, k))
        """
        try:
            from scipy.sparse.linalg import eigsh
            eigenvalues, eigenvectors = eigsh(
                self._A, k=k, which="SM",
                tol=self._tol, maxiter=self._max_iter,
                M=self._precond,
            )
            # Sort by ascending eigenvalue
            order = np.argsort(eigenvalues)
            return eigenvalues[order], eigenvectors[:, order]
        except Exception as exc:
            logger.warning(f"Eigensolver failed: {exc}")
            n = self._A.shape[0]
            return np.zeros(k), np.zeros((n, k))


# ---------------------------------------------------------------------------
# Convenience helpers used by other engine modules
# ---------------------------------------------------------------------------

def make_solver(A, params=None) -> SparseGPUSolver:
    """Factory: create a SparseGPUSolver with settings from QFParams.

    Parameters
    ----------
    A : scipy.sparse matrix
        The coefficient matrix.
    params : QFParams-like object, optional
        If provided, reads .use_gpu (bool) to set the backend.
    """
    use_gpu = None
    if params is not None:
        use_gpu = bool(getattr(params, "use_gpu", False))
    return SparseGPUSolver(A, use_gpu=use_gpu)


def gpu_status_string() -> str:
    """Short status string for the UI info panel."""
    info = detect_gpu()
    if not info.available:
        return "GPU: not detected (CPU solver active)"
    return f"GPU: {info.device_name} [{info.backend.upper()}] ~{info.estimated_speedup:.1f}× speedup"
