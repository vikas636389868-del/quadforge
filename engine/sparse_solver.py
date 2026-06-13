"""Sparse cross-field solver for QuadForge.

Provides a sparse-matrix-based connection Laplacian assembly and
eigensolver that can handle meshes with 50K+ faces, replacing the
dense O(F²) implementation in field.py for large meshes.

Uses scipy.sparse when available, falling back to a hand-rolled
sparse COO/CSR implementation backed by numpy for environments
(like Blender's embedded Python) where scipy may not be installed.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Set, Tuple


# ======================================================================
# Sparse matrix support — try scipy, fall back to manual
# ======================================================================

_HAS_SCIPY = False
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    _HAS_SCIPY = True
except ImportError:
    pass


class _SparseCOO:
    """Minimal sparse COO matrix for when scipy is unavailable."""

    def __init__(self, n: int, dtype=np.complex128):
        self.n = n
        self.rows: List[int] = []
        self.cols: List[int] = []
        self.vals: List[complex] = []
        self.dtype = dtype

    def add(self, i: int, j: int, val):
        self.rows.append(i)
        self.cols.append(j)
        self.vals.append(val)

    def to_dense(self) -> np.ndarray:
        M = np.zeros((self.n, self.n), dtype=self.dtype)
        for r, c, v in zip(self.rows, self.cols, self.vals):
            M[r, c] += v
        return M


def assemble_sparse_connection_laplacian(
    num_faces: int,
    edge_to_faces: Dict[Tuple[int, int], List[int]],
    transport_angles: Dict[Tuple[int, int], float],
    edge_weights: Dict[Tuple[int, int], float],
    constraint_faces: Optional[Dict[int, float]] = None,
    constraint_strength: float = 100.0,
) -> any:
    """Assemble the connection Laplacian as a sparse matrix.

    Returns a scipy.sparse.csr_matrix if scipy is available,
    otherwise returns a dense numpy array (falling back for small cases).

    Parameters
    ----------
    num_faces : number of faces (matrix dimension)
    edge_to_faces : edge key → face list
    transport_angles : edge key → parallel transport angle φ
    edge_weights : edge key → weight w
    constraint_faces : face_idx → target angle (for penalty terms)
    constraint_strength : penalty weight for constraints

    Returns
    -------
    L : sparse or dense complex matrix of shape (F, F)
    """
    if _HAS_SCIPY and num_faces >= 1000:
        return _assemble_scipy(
            num_faces, edge_to_faces, transport_angles,
            edge_weights, constraint_faces, constraint_strength,
        )
    else:
        return _assemble_dense(
            num_faces, edge_to_faces, transport_angles,
            edge_weights, constraint_faces, constraint_strength,
        )


def _assemble_scipy(
    num_faces, edge_to_faces, transport_angles,
    edge_weights, constraint_faces, constraint_strength,
):
    """Assemble using scipy sparse matrices."""
    rows, cols, vals = [], [], []

    # Diagonal accumulator
    diag = np.zeros(num_faces, dtype=np.complex128)

    for edge_key, face_list in edge_to_faces.items():
        if len(face_list) != 2:
            continue
        fi, fj = face_list[0], face_list[1]
        phi = transport_angles.get(edge_key, 0.0)
        w = edge_weights.get(edge_key, 1.0)

        rot = np.exp(4j * phi)

        diag[fi] += w
        diag[fj] += w

        rows.append(fi)
        cols.append(fj)
        vals.append(-w * rot)

        rows.append(fj)
        cols.append(fi)
        vals.append(-w * np.conj(rot))

    # Add constraint penalties to diagonal
    if constraint_faces:
        for fi, _ in constraint_faces.items():
            if 0 <= fi < num_faces:
                diag[fi] += constraint_strength

    # Add diagonal entries
    for i in range(num_faces):
        rows.append(i)
        cols.append(i)
        vals.append(diag[i])

    L = sp.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(num_faces, num_faces),
        dtype=np.complex128,
    )
    return L


def _assemble_dense(
    num_faces, edge_to_faces, transport_angles,
    edge_weights, constraint_faces, constraint_strength,
):
    """Assemble as dense matrix (fallback for small meshes or no scipy)."""
    L = np.zeros((num_faces, num_faces), dtype=np.complex128)

    for edge_key, face_list in edge_to_faces.items():
        if len(face_list) != 2:
            continue
        fi, fj = face_list[0], face_list[1]
        phi = transport_angles.get(edge_key, 0.0)
        w = edge_weights.get(edge_key, 1.0)

        rot = np.exp(4j * phi)

        L[fi, fi] += w
        L[fj, fj] += w
        L[fi, fj] -= w * rot
        L[fj, fi] -= w * np.conj(rot)

    if constraint_faces:
        for fi, _ in constraint_faces.items():
            if 0 <= fi < num_faces:
                L[fi, fi] += constraint_strength

    return L


def solve_sparse_eigenvector(
    L,
    constraint_faces: Optional[Dict[int, complex]] = None,
    constraint_strength: float = 100.0,
    num_faces: int = 0,
) -> np.ndarray:
    """Solve for the smallest eigenvector of the connection Laplacian.

    Uses scipy.sparse.linalg.eigsh (Lanczos) for large sparse matrices,
    np.linalg.eigh for dense, with shift-invert for better convergence.

    Parameters
    ----------
    L : sparse or dense connection Laplacian
    constraint_faces : face_idx → target complex value
    constraint_strength : penalty weight
    num_faces : matrix dimension (for validation)

    Returns
    -------
    x : complex eigenvector of length F
    """
    if _HAS_SCIPY and sp.issparse(L):
        return _solve_scipy(L, constraint_faces, constraint_strength)
    else:
        return _solve_dense(L, constraint_faces, constraint_strength)


def _solve_scipy(L, constraint_faces, constraint_strength):
    """Solve using scipy shift-invert Lanczos."""
    n = L.shape[0]

    # Build RHS for constrained system
    rhs = np.zeros(n, dtype=np.complex128)
    has_rhs = False

    if constraint_faces:
        for fi, target_u in constraint_faces.items():
            if 0 <= fi < n:
                rhs[fi] = constraint_strength * target_u
                has_rhs = True

    if has_rhs:
        # Soft-constraint linear solve: (L + εI) x = rhs
        try:
            L_reg = L + sp.eye(n, dtype=np.complex128) * 1e-8
            x = spla.spsolve(L_reg, rhs)
        except Exception:
            try:
                result = spla.lsqr(L, rhs)
                x = result[0]
            except Exception:
                x = np.ones(n, dtype=np.complex128)
    else:
        # Pure eigenproblem — find the eigenvector of smallest magnitude.
        # The connection Laplacian is Hermitian (L[i,j] = conj(L[j,i]))
        # by construction. scipy's eigsh handles complex Hermitian
        # matrices directly. On a purely-flat mesh all transport angles
        # are 0 and the matrix is real symmetric, in which case we can
        # speed things up by passing the real view.
        x = None
        L_for_eig = L
        if np.allclose(L.data.imag, 0.0, atol=1e-12):
            L_for_eig = L.real.astype(np.float64)

        # Shift-invert Lanczos around σ = small positive to target
        # the smallest eigenvalue robustly.
        for sigma in (1e-6, 1e-5, 1e-4):
            try:
                eigenvalues, eigenvectors = spla.eigsh(
                    L_for_eig,
                    k=1,
                    sigma=sigma,
                    which="LM",
                    maxiter=1000,
                    tol=1e-7,
                )
                x = eigenvectors[:, 0].astype(np.complex128)
                break
            except Exception:
                continue

        if x is None:
            # Fallback: plain eigsh without shift-invert
            try:
                eigenvalues, eigenvectors = spla.eigsh(
                    L_for_eig, k=6, which="SM", maxiter=2000,
                )
                x = eigenvectors[:, 0].astype(np.complex128)
            except Exception:
                x = _sparse_inverse_iteration(L, n)

    # Normalise
    mags = np.abs(x)
    max_mag = np.max(mags)
    if max_mag > 1e-15:
        x /= max_mag

    return x


def _sparse_inverse_iteration(L, n, max_iter=200):
    """Sparse inverse iteration for smallest eigenvector."""
    sigma = 1e-6
    A = L + sp.eye(n, dtype=np.complex128) * sigma

    x = np.random.RandomState(42).randn(n).astype(np.complex128)
    x /= np.linalg.norm(x)

    try:
        lu = spla.splu(A.tocsc())
        for _ in range(max_iter):
            y = lu.solve(x)
            norm = np.linalg.norm(y)
            if norm < 1e-15:
                break
            x = y / norm
    except Exception:
        pass

    return x


def _solve_dense(L, constraint_faces, constraint_strength):
    """Solve using dense numpy (for small matrices)."""
    n = L.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.complex128)

    rhs = np.zeros(n, dtype=np.complex128)
    has_rhs = False

    if constraint_faces:
        for fi, target_u in constraint_faces.items():
            if 0 <= fi < n:
                rhs[fi] = constraint_strength * target_u
                has_rhs = True

    if has_rhs:
        try:
            L_reg = L + 1e-10 * np.eye(n, dtype=np.complex128)
            x = np.linalg.solve(L_reg, rhs)
        except np.linalg.LinAlgError:
            x = np.ones(n, dtype=np.complex128)
    else:
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            x = eigenvectors[:, 0]
        except np.linalg.LinAlgError:
            x = np.ones(n, dtype=np.complex128)

    mags = np.abs(x)
    max_mag = np.max(mags)
    if max_mag > 1e-15:
        x /= max_mag

    return x


def sparse_poisson_solve(
    A_entries: List[Tuple[int, int, float]],
    rhs: np.ndarray,
    num_verts: int,
    pin_vertex: int = 0,
) -> np.ndarray:
    """Solve a Poisson system using sparse matrices.

    Used by the parametrization stage for meshes too large for
    dense np.linalg.solve.

    Parameters
    ----------
    A_entries : list of (row, col, value) triplets
    rhs : right-hand side vector
    num_verts : system dimension
    pin_vertex : vertex to pin (translation fix)

    Returns
    -------
    x : solution vector
    """
    if _HAS_SCIPY and num_verts >= 1000:
        rows, cols, vals = zip(*A_entries) if A_entries else ([], [], [])
        A = sp.csr_matrix(
            (list(vals), (list(rows), list(cols))),
            shape=(num_verts, num_verts),
            dtype=np.float64,
        )
        # Pin vertex
        A[pin_vertex, :] = 0
        A[:, pin_vertex] = 0
        A[pin_vertex, pin_vertex] = 1.0
        rhs = rhs.copy()
        rhs[pin_vertex] = 0.0

        # Add regularisation
        A += sp.eye(num_verts, dtype=np.float64) * 1e-8

        try:
            x = spla.spsolve(A.tocsc(), rhs)
        except Exception:
            try:
                result = spla.lsqr(A, rhs)
                x = result[0]
            except Exception:
                x = np.zeros(num_verts, dtype=np.float64)
        return x
    else:
        # Dense fallback
        A = np.zeros((num_verts, num_verts), dtype=np.float64)
        for r, c, v in A_entries:
            A[r, c] += v
        A += 1e-8 * np.eye(num_verts)
        A[pin_vertex, :] = 0
        A[:, pin_vertex] = 0
        A[pin_vertex, pin_vertex] = 1.0
        rhs = rhs.copy()
        rhs[pin_vertex] = 0.0

        try:
            return np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            return np.zeros(num_verts, dtype=np.float64)
