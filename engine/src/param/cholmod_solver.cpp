/**
 * cholmod_solver.cpp — CHOLMOD-backed sparse Cholesky SPD solver.
 *
 * The entire file body is conditionally compiled.  When QF_HAS_CHOLMOD is
 * NOT defined (CMake didn't find SuiteSparse at configure time), this
 * translation unit becomes empty and contributes nothing to libquadforge —
 * the engine's CMake globs all .cpp under src/ unconditionally, so guarding
 * the body is the simplest way to keep the build working on systems without
 * SuiteSparse installed.
 */

#include "cholmod_solver.h"

#ifdef QF_HAS_CHOLMOD

#include <cstring>

namespace qf {

// -----------------------------------------------------------------------
// Construction / destruction
// -----------------------------------------------------------------------

CholmodSPDSolver::CholmodSPDSolver() {
    // cholmod_start returns TRUE on success (per cholmod.h convention).  If
    // it fails we mark the solver as unstarted and every later call becomes
    // a no-op returning false.
    if (cholmod_start(&common_)) {
        started_ = true;
        // Prefer the LL' (Cholesky) form for cheaper triangular solves —
        // LDL' adds a divide per row for no advantage when we know the
        // system is SPD.
        common_.final_ll      = 1;
        common_.print         = 0;       // silent; we surface failure via ok()
        common_.error_handler = nullptr;
    }
}

CholmodSPDSolver::~CholmodSPDSolver() {
    if (started_) {
        if (factor_ != nullptr) {
            cholmod_free_factor(&factor_, &common_);
            factor_ = nullptr;
        }
        cholmod_finish(&common_);
    }
}

// -----------------------------------------------------------------------
// factorize
// -----------------------------------------------------------------------

bool CholmodSPDSolver::factorize(const Eigen::SparseMatrix<double>& A) {
    ok_ = false;
    if (!started_)           return false;
    if (A.rows() != A.cols()) return false;
    if (A.rows() == 0)        return false;

    // Release any factor from a previous call.
    if (factor_ != nullptr) {
        cholmod_free_factor(&factor_, &common_);
        factor_ = nullptr;
    }

    // Extract the upper triangle into an Eigen-owned compressed matrix.
    // We hand CHOLMOD a stype = 1 view that aliases this storage, so the
    // object must outlive every subsequent solve() call — hence it's a
    // member, not a local.
    upper_ = A.triangularView<Eigen::Upper>();
    upper_.makeCompressed();

    // Build a cholmod_sparse view on the Eigen CSC storage (no copy).
    // Eigen's default SparseMatrix<double> uses int32_t for StorageIndex and
    // column-major layout, which matches CHOLMOD_INT + standard CSC exactly.
    cholmod_sparse A_view;
    std::memset(&A_view, 0, sizeof(A_view));
    A_view.nrow   = static_cast<size_t>(upper_.rows());
    A_view.ncol   = static_cast<size_t>(upper_.cols());
    A_view.nzmax  = static_cast<size_t>(upper_.nonZeros());
    A_view.p      = const_cast<int*>(upper_.outerIndexPtr());
    A_view.i      = const_cast<int*>(upper_.innerIndexPtr());
    A_view.x      = const_cast<double*>(upper_.valuePtr());
    A_view.nz     = nullptr;           // packed matrix — no per-col nz array
    A_view.z      = nullptr;           // real only — no imaginary part
    A_view.stype  = 1;                 // symmetric, upper triangle stored
    A_view.itype  = CHOLMOD_INT;       // int32 indices
    A_view.xtype  = CHOLMOD_REAL;
    A_view.dtype  = CHOLMOD_DOUBLE;
    A_view.sorted = 1;                 // Eigen compressed matrices are sorted
    A_view.packed = 1;

    // Symbolic analysis (column ordering + supernodal symbolic).  Returns a
    // freshly allocated cholmod_factor, or NULL on failure.
    //
    // BUG FIX (Bug 1-analyze — param/cholmod): Use != 0 instead of < 0.
    // The previous check `< 0` only caught hard errors (negative status).
    // CHOLMOD may set common_.status > 0 during cholmod_analyze to report
    // ordering warnings (e.g. CHOLMOD_NOT_POSDEF detected during AMD/COLAMD
    // ordering).  Those warnings indicate the matrix is likely not SPD, which
    // means the subsequent cholmod_factorize will also warn or fail.  Treating
    // them as success allowed a corrupt/useless factor to pass into factorize()
    // and ultimately into solve().  Consistent with the != 0 check already used
    // for cholmod_factorize() at the bottom of this function.
    common_.status = 0;
    factor_ = cholmod_analyze(&A_view, &common_);
    if (factor_ == nullptr || common_.status != 0) {
        if (factor_) {
            cholmod_free_factor(&factor_, &common_);
            factor_ = nullptr;
        }
        return false;
    }

    // Numeric factorisation.  Returns TRUE on success, FALSE on failure.
    // common_.status > 0 indicates a warning (most commonly CHOLMOD_NOT_POSDEF
    // when the matrix turned out indefinite mid-factorisation); we treat that
    // as a hard failure because the Poisson system is SPD by construction
    // and any warning means something upstream is wrong.
    if (!cholmod_factorize(&A_view, factor_, &common_) || common_.status != 0) {
        cholmod_free_factor(&factor_, &common_);
        factor_ = nullptr;
        return false;
    }

    ok_ = true;
    return true;
}

// -----------------------------------------------------------------------
// solve
// -----------------------------------------------------------------------

Eigen::VectorXd CholmodSPDSolver::solve(const Eigen::VectorXd& b) {
    Eigen::VectorXd x;
    if (!ok_ || factor_ == nullptr || b.size() == 0) {
        ok_ = false;
        return x;
    }
    if (static_cast<size_t>(b.size()) !=
        static_cast<size_t>(upper_.rows())) {
        ok_ = false;
        return x;
    }

    // Build a cholmod_dense view on b's data (no copy).  d == nrow because
    // a VectorXd is contiguous with unit stride.
    cholmod_dense b_view;
    std::memset(&b_view, 0, sizeof(b_view));
    b_view.nrow  = static_cast<size_t>(b.size());
    b_view.ncol  = 1;
    b_view.nzmax = static_cast<size_t>(b.size());
    b_view.d     = static_cast<size_t>(b.size());
    b_view.x     = const_cast<double*>(b.data());
    b_view.z     = nullptr;
    b_view.xtype = CHOLMOD_REAL;
    b_view.dtype = CHOLMOD_DOUBLE;

    common_.status = 0;
    cholmod_dense* x_dense =
        cholmod_solve(CHOLMOD_A, factor_, &b_view, &common_);
    // BUG FIX (Bug 1 — param/cholmod): Use != 0 instead of < 0.
    // CHOLMOD status > 0 indicates a warning (e.g. CHOLMOD_NOT_POSDEF during
    // solve).  The previous check `< 0` only caught hard errors; a positive
    // status meant a corrupted x_dense was silently accepted.  factorize()
    // already uses `!= 0` — this brings solve() into consistency with it.
    if (x_dense == nullptr || common_.status != 0) {
        if (x_dense != nullptr) cholmod_free_dense(&x_dense, &common_);
        ok_ = false;
        return x;
    }

    // Copy result back into an Eigen vector (x_dense is CHOLMOD-owned).
    x.resize(b.size());
    std::memcpy(x.data(), x_dense->x, sizeof(double) * static_cast<size_t>(b.size()));
    cholmod_free_dense(&x_dense, &common_);
    return x;
}

} // namespace qf

#endif // QF_HAS_CHOLMOD
