#pragma once
/**
 * cholmod_solver.h — private wrapper around CHOLMOD's supernodal sparse
 * Cholesky for symmetric positive-definite systems.
 *
 * This header is compiled-empty unless the CMake build defines QF_HAS_CHOLMOD
 * (which FindSuiteSparse.cmake attaches to the quadforge target when
 * SuiteSparse is located at configure time).  That way the engine still
 * compiles cleanly on systems without SuiteSparse installed — it just falls
 * through to Eigen's SimplicialLDLT / SparseLU path in poisson.cpp.
 *
 * Not part of the public engine API.  Lives under src/param/ and is #included
 * only from poisson.cpp.
 */

#ifndef QUADFORGE_PARAM_CHOLMOD_SOLVER_H
#define QUADFORGE_PARAM_CHOLMOD_SOLVER_H

#ifdef QF_HAS_CHOLMOD

#include <Eigen/Sparse>
#include <Eigen/Dense>

// CHOLMOD public header.  FindSuiteSparse.cmake adds the directory that
// physically contains cholmod.h to the quadforge target's include path, so
// this include resolves against both the vendored headers in
// third_party/suitesparse/include/suitesparse/ and any system install
// (libsuitesparse-dev on Linux, brew's suite-sparse on macOS, vcpkg on
// Windows).
#include <cholmod.h>

namespace qf {

/**
 * RAII wrapper around CHOLMOD for symmetric positive-definite systems.
 *
 * Usage pattern:
 *
 *     CholmodSPDSolver solver;
 *     if (!solver.factorize(L_sym)) { fallback(); }
 *     Eigen::VectorXd xu = solver.solve(bu);
 *     Eigen::VectorXd xv = solver.solve(bv);
 *     if (!solver.ok()) { fallback(); }
 *
 * The wrapper is non-copyable and non-movable to keep the cholmod_common
 * lifetime trivially tied to the solver object.  All CHOLMOD resources
 * (cholmod_common, cholmod_factor, and the internal upper-triangle copy of
 * the matrix) are released in the destructor.
 *
 * Pass in a symmetric matrix stored with BOTH halves — the wrapper extracts
 * the upper triangle itself via Eigen::triangularView<Upper>() and hands
 * CHOLMOD a cholmod_sparse view with stype = 1.
 */
class CholmodSPDSolver {
public:
    CholmodSPDSolver();
    ~CholmodSPDSolver();

    CholmodSPDSolver(const CholmodSPDSolver&)            = delete;
    CholmodSPDSolver& operator=(const CholmodSPDSolver&) = delete;
    CholmodSPDSolver(CholmodSPDSolver&&)                 = delete;
    CholmodSPDSolver& operator=(CholmodSPDSolver&&)      = delete;

    /**
     * Analyse the sparsity pattern of A and perform the numeric factorisation.
     * A must be square and symmetric positive definite.
     *
     * @return true on success; false on any CHOLMOD error (non-SPD matrix,
     *              allocation failure, CHOLMOD not started, etc.).
     */
    bool factorize(const Eigen::SparseMatrix<double>& A);

    /**
     * Solve A * x = b using the stored factor.  factorize() must have
     * returned true before this is called.  Returns an empty vector on
     * failure and sets ok() to false.
     */
    Eigen::VectorXd solve(const Eigen::VectorXd& b);

    /** True if the solver is ready and the last operation succeeded. */
    bool ok() const { return ok_ && factor_ != nullptr; }

private:
    cholmod_common  common_{};
    cholmod_factor* factor_{nullptr};

    // Eigen-owned upper-triangle copy of the input matrix.  We must keep this
    // alive for the lifetime of the solver because the cholmod_sparse view we
    // hand to CHOLMOD aliases these pointers directly (no copy).
    Eigen::SparseMatrix<double> upper_;

    bool started_{false};
    bool ok_{false};
};

} // namespace qf

#endif // QF_HAS_CHOLMOD
#endif // QUADFORGE_PARAM_CHOLMOD_SOLVER_H
