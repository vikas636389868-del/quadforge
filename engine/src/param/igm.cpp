/**
 * igm.cpp — Integer Grid Maps (IGM) parametrization.
 *
 * Implements Bommes et al. (2013) "Integer-Grid Maps for Reliable Quad
 * Meshing."  Key improvement over MIQ: period jumps at seam cuts are
 * rounded GLOBALLY using a Union-Find spanning-tree approach that
 * enforces consistent integer transitions across every seam edge.
 *
 * Algorithm:
 *   1. Solve the continuous Poisson system (same as MIQ start).
 *   2. Compute the raw (real-valued) period jumps at each seam edge.
 *   3. Round period jumps globally using a spanning-tree traversal of
 *      the seam graph (Union-Find with integer offset propagation).
 *   4. Re-solve Poisson with the chosen integer jumps enforced as
 *      Dirichlet constraints → parametrization honours exact integers.
 *   5. Strict global snap: snap ALL vertices to nearest integer grid.
 *
 * References:
 *   Bommes et al. (2013), "Integer-Grid Maps for Reliable Quad Meshing,"
 *   ACM Trans. Graph. 32(4) (SIGGRAPH 2013).
 */

#include "../../include/quadforge/param/igm.h"
#include "../../include/quadforge/accel/omp_utils.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qf {

// -----------------------------------------------------------------------
// Union-Find with per-node float offsets (for period jump propagation)
// -----------------------------------------------------------------------

struct UFOffset {
    std::vector<int>    parent;
    std::vector<int>    rank;
    std::vector<double> off_u;   // cumulative U offset to root
    std::vector<double> off_v;   // cumulative V offset to root

    explicit UFOffset(int n)
        : parent(n), rank(n, 0), off_u(n, 0.0), off_v(n, 0.0)
    {
        std::iota(parent.begin(), parent.end(), 0);
    }

    // Find root with path compression; accumulates offsets along the way.
    int find(int x, double& acc_u, double& acc_v) {
        if (parent[x] == x) { acc_u = 0; acc_v = 0; return x; }
        double pu = 0, pv = 0;
        int root = find(parent[x], pu, pv);
        // Update path: x's offset relative to root
        off_u[x] += pu;
        off_v[x] += pv;
        parent[x] = root;
        acc_u = off_u[x];
        acc_v = off_v[x];
        return root;
    }

    // Unite two components.
    // The edge (a → b) has a target integer jump (ju, jv), meaning:
    //   U[b] - U[a] should equal ju  (and similarly for V)
    // Returns false if merging would create a contradiction > tol.
    bool unite(int a, int b, double ju, double jv, double tol = 0.5) {
        double oa_u = 0, oa_v = 0, ob_u = 0, ob_v = 0;
        int ra = find(a, oa_u, oa_v);
        int rb = find(b, ob_u, ob_v);

        if (ra == rb) {
            // Check consistency: U[b] - U[a] = ob_u - oa_u  (should ≈ ju)
            double existing_jump_u = ob_u - oa_u;
            double existing_jump_v = ob_v - oa_v;
            return (std::abs(existing_jump_u - ju) < tol &&
                    std::abs(existing_jump_v - jv) < tol);
        }

        // Merge rb into ra.
        // We need: off[b] relative to ra  = oa_u + ju
        // Currently off[b] relative to rb = ob_u  (= 0 if rb is root)
        // So the offset of rb's root relative to ra's root = oa_u + ju - ob_u
        if (rank[ra] < rank[rb]) {
            std::swap(ra, rb);
            std::swap(oa_u, ob_u);
            std::swap(oa_v, ob_v);
            // Flipped: now we need U[a] - U[b] = -ju
            ju = -ju;  jv = -jv;
        }

        parent[rb] = ra;
        // offset of rb_root relative to ra_root:
        //   offset[rb_root] = oa_u + ju - ob_u
        off_u[rb] = oa_u + ju - ob_u;
        off_v[rb] = oa_v + jv - ob_v;

        if (rank[ra] == rank[rb]) ++rank[ra];
        return true;
    }

    // Get the offset of vertex x relative to its root.
    void get_offset(int x, double& out_u, double& out_v) {
        find(x, out_u, out_v);
    }
};

// -----------------------------------------------------------------------
// verify_jump_consistency
// -----------------------------------------------------------------------

int verify_jump_consistency(
    const HalfEdgeMesh&     mesh,
    const EdgeSet&          seam_edges,
    const std::vector<int>& jump_u,
    const std::vector<int>& jump_v,
    double                  tol)
{
    int violations = 0;
    // BUG FIX (Bug 3 — param/igm): `nv` was declared but never used; vertex
    // indices in seam_edges were never bounds-checked, so an out-of-range index
    // (from a corrupt CombingResult or mismatched mesh) would silently corrupt
    // the vertex_sum map rather than being detected.  Added a per-edge guard.
    int nv = mesh.num_vertices();

    // BUG FIX (B1 — param/igm v110): Add bounds check on jump_u / jump_v
    // vectors.  The function accepts seam_edges, jump_u, and jump_v as
    // independent arguments with no compile-time constraint on their sizes.
    // If a caller passes vectors of mismatched length (e.g. seam_edges has
    // extra entries that were padded by engine.cpp and carry no jump data),
    // accessing jump_u[k] / jump_v[k] for k >= jump_u.size() is undefined
    // behaviour — a hard crash or silent memory corruption.
    //
    // Fix: cap the loop to the minimum of the three sizes, and count any
    // entry beyond that cap as a violation (the jump is unknown / missing).
    const int n_ju   = static_cast<int>(jump_u.size());
    const int n_jv   = static_cast<int>(jump_v.size());
    const int n_seam = static_cast<int>(seam_edges.size());

    // Any seam edge whose jump data is absent is a violation.
    violations += std::max(0, n_seam - std::min(n_ju, n_jv));

    const int n_valid = std::min({ n_seam, n_ju, n_jv });

    // BUG FIX (P6 — param/igm v112): Remove dead jump_map construction.
    //
    // The previous code built a std::unordered_map<long long, std::pair<int,int>>
    // jump_map (O(n_valid) hash inserts) and then NEVER queried it — the
    // vertex_sum loop below always accessed jump_u[k] / jump_v[k] directly by
    // index.  jump_map was entirely dead code: allocated, populated, and silently
    // discarded without contributing anything to the violation count or the
    // vertex_sum computation.
    //
    // The only useful work that occurred inside the old jump_map loop was the
    // bounds-check that incremented violations for malformed edges.  That check
    // has been merged into the vertex_sum loop below so that malformed edges are
    // still counted as violations (behaviour preserved).
    //
    // Memory savings on a mesh with 1M seam edges: ~48 MB (unordered_map node
    // overhead) + hash table bucket array.  Time savings: O(n_valid) hash inserts
    // and the load-factor rehash that follows.

    // For each seam vertex, sum the jumps in its one-ring of seam edges.
    // A consistent assignment has zero net jump for interior seam vertices.
    std::unordered_map<int, std::pair<double,double>> vertex_sum;
    for (int k = 0; k < n_valid; ++k) {
        int lo = seam_edges[k].first, hi = seam_edges[k].second;
        // Bounds-check: count malformed edges as violations (same semantics as
        // the old jump_map loop that this replaces — see P6 fix above).
        if (lo < 0 || lo >= nv || hi < 0 || hi >= nv) {
            ++violations;
            continue;
        }
        vertex_sum[lo].first  += jump_u[k];
        vertex_sum[lo].second += jump_v[k];
        vertex_sum[hi].first  -= jump_u[k];
        vertex_sum[hi].second -= jump_v[k];
    }

    for (auto& [v, sums] : vertex_sum) {
        // BUG FIX (P5 — param/igm v111): Replace no-op near-integer check.
        //
        // Root cause: jump_u / jump_v are declared as std::vector<int>, so
        // every accumulated sum (sums.first / sums.second) is already an exact
        // integer stored in a double.  The original check
        //
        //     fu = sums.first - std::round(sums.first);    // always 0.0
        //     if (std::abs(fu) > tol) ...                  // always false
        //
        // can NEVER fire — std::round(x) == x for any x that is an exact
        // integer in double precision.  The entire consistency verification was
        // a dead-code no-op that always returned 0 violations regardless of
        // actual jump inconsistency in the input data.
        //
        // Correct semantics:
        //   • Interior seam vertex: net jump sum should be exactly 0.
        //   • ±1/4 singularity vertex: net jump sum may be ±1 in one
        //     coordinate (because the 4-RoSy field has a single ±90° wrap
        //     around the vertex).  We allow |net| == 1 to avoid false positives
        //     for valid singularity configurations.
        //   • Any |net| > 1, or a net value that is inconsistent (e.g. both
        //     coordinates non-zero, or |net| > 1 in either), indicates a
        //     genuine inconsistency: duplicate seam edges, sign errors in the
        //     combing rotation assignments, or a corrupt CombingResult.
        //
        // The tol parameter is retained in the function signature for API
        // compatibility but is no longer used, since the sums are integers and
        // no floating-point tolerance is meaningful here.
        (void)tol;  // integer sums — no floating-point tolerance needed

        int net_u = static_cast<int>(sums.first);
        int net_v = static_cast<int>(sums.second);

        // Allow net = 0 (interior vertex, consistent) and |net| = 1 in exactly
        // one coordinate (±1/4 singularity, one 90° wrap).
        // Flag everything else as a violation.
        bool u_ok = (net_u == 0 || net_u == 1 || net_u == -1);
        bool v_ok = (net_v == 0 || net_v == 1 || net_v == -1);
        // Both non-zero simultaneously → inconsistency (no pure singularity
        // creates a non-zero jump in both U and V at the same vertex).
        bool both_nonzero = (net_u != 0 && net_v != 0);
        if (!u_ok || !v_ok || both_nonzero)
            ++violations;
    }

    return violations;
}

// -----------------------------------------------------------------------
// compute_igm_parametrization
// -----------------------------------------------------------------------

UVParam compute_igm_parametrization(
    const HalfEdgeMesh&    mesh,
    const CombingResult&   combing,
    const Eigen::VectorXd& sizing,
    const IGMParams&       params)
{
    int nv = mesh.num_vertices();
    UVParam uv;
    uv.U.setZero(nv);
    uv.V.setZero(nv);
    uv.seam_edges = combing.seam_edges;
    // BUG FIX (v89): Transfer period jumps into UVParam so seam_utils can
    // build a PeriodJumpMap for seam-aware isoline tracing. Previously,
    // period_jumps lived only in CombingResult and were never forwarded.
    //
    // BUG FIX (v109 — param/igm): Pad period_jumps to match seam_edges size.
    //
    // engine.cpp appends extra seam-cut edges (from compute_seam_cut_from_mesh)
    // to combing.seam_edges AFTER comb_field() returns, so:
    //   combing.seam_edges.size()   = original + extras
    //   combing.period_jumps.size() = original only
    //
    // The previous code built uv.period_jumps only from combing.period_jumps,
    // leaving uv.period_jumps.size() < uv.seam_edges.size() whenever engine.cpp
    // appended extra seam edges.  This violated the PRECONDITION in isolines.h
    // (seam_edges and period_jumps must be parallel arrays of equal length) and
    // caused seam_utils::build_period_jump_map() to silently return an empty map,
    // making extract_quads_from_isolines_seam_aware() fall back to the seam-blind
    // variant — producing T-junctions at UV seam crossings for all IGM solves.
    //
    // The equivalent fix was already applied to miq.cpp (v108 param/miq fix).
    // Extra seam-cut edges from Dijkstra / feature seams carry no combing-derived
    // period jump.  We pad them with (0, 0), which build_period_jump_map() treats
    // as "no UV shift" — the conservative correct behaviour for those edges.
    //
    // Postcondition: uv.seam_edges.size() == uv.period_jumps.size().
    {
        const int n_seams = static_cast<int>(combing.seam_edges.size());
        const int n_jumps = static_cast<int>(combing.period_jumps.size());
        uv.period_jumps.clear();
        uv.period_jumps.reserve(static_cast<size_t>(n_seams));
        for (int k = 0; k < n_seams; ++k) {
            if (k < n_jumps) {
                int pj = combing.period_jumps[k];
                int du = static_cast<int16_t>((pj >> 16) & 0xFFFF);
                int dv = static_cast<int16_t>( pj        & 0xFFFF);
                uv.period_jumps.emplace_back(du, dv);
            } else {
                // Extra seam-cut edge appended by engine.cpp after comb_field():
                // no combing-derived period jump → pad with (0, 0).
                uv.period_jumps.emplace_back(0, 0);
            }
        }
    }

    if (nv == 0) return uv;

    // BUG FIX (v24): removed omp_set_num_threads() — global setter.
    // Thread count is passed per-region via parallel_for() num_threads() clause.

    // BUG FIX (Bug 5 — param/miq+igm): Find the first NON-SEAM vertex to
    // use as the translation-DOF pin in LDLT fallbacks.  Pinning vertex 0
    // unconditionally risks a contradiction when vertex 0 is a seam vertex
    // with an active integer jump constraint (the pin forces U[0]=0 while the
    // seam constraint demands U[0] = U[neighbour] + k for some non-zero k).
    std::unordered_set<int> seam_vset_igm;
    seam_vset_igm.reserve(combing.seam_edges.size() * 2);
    for (auto& [s0, s1] : combing.seam_edges) {
        seam_vset_igm.insert(s0);
        seam_vset_igm.insert(s1);
    }
    int pin_vertex_igm = 0;
    for (int i = 0; i < nv; ++i) {
        if (!seam_vset_igm.count(i)) { pin_vertex_igm = i; break; }
    }

    // ----------------------------------------------------------------
    // Step 1: Continuous Poisson solve (shared with MIQ)
    // ----------------------------------------------------------------
    SparseMat A_u; VecX b_u;
    build_poisson_system(mesh, combing.face_angles, sizing, A_u, b_u);

    // V-direction: same Laplacian (geometry-only), RHS rotated by π/2.
    // BUG FIX (B5 — param/igm v110): Same fix as B4 in miq.cpp.
    // A_v is geometrically identical to A_u; only the RHS differs.
    // Build V-RHS via build_poisson_system and discard the redundant LHS.
    std::vector<double> v_angles(combing.face_angles.size());
    for (size_t i = 0; i < combing.face_angles.size(); ++i)
        v_angles[i] = combing.face_angles[i] + M_PI / 2.0;

    SparseMat A_v_discard; VecX b_v;
    build_poisson_system(mesh, v_angles, sizing, A_v_discard, b_v);
    A_v_discard.resize(0, 0);   // release memory — A_u is reused below

    // A_v == A_u; alias for clarity in the solve blocks.
    const SparseMat& A_v = A_u;

    // Solve with ConjugateGradient (reuse factorisation for U and V)
    Eigen::ConjugateGradient<SparseMat,
                             Eigen::Lower|Eigen::Upper> cg;
    cg.setMaxIterations(params.max_cg_iterations);
    cg.setTolerance(params.cg_tolerance);

    // BUG FIX (v72 — igm Step 1 fallback):
    //
    // The original code had NO fallback if cg.compute() returned non-Success.
    // In that case uv.U and uv.V silently stayed zero.  The Union-Find rounding
    // in Step 3 then computed all raw jumps as 0.0, and the penalty re-solve in
    // Step 4 enforced "all seam jumps = 0" — producing a collapsed UV layout
    // and meaningless quad extraction downstream.
    //
    // The fix mirrors the identical pattern already in miq.cpp: on CG failure,
    // fall through to Eigen SimplicialLDLT with a soft Dirichlet pin on the
    // first non-seam vertex (adds +1e4 to the diagonal entry to eliminate the
    // translation null-space without hard-zeroing the row).
    //
    // BUG FIX (Bug 4 — param/igm): In Eigen, cg.solve() updates cg.info()
    // to Success or NoConvergence.  Because cg.compute(A_v) was called right
    // after cg.solve(b_u), the U solve convergence result was overwritten
    // before it could be inspected.  Fix: capture and act on solve convergence
    // immediately after each solve() call, before the next compute() resets it.

    cg.compute(A_u);
    if (cg.info() == Eigen::Success) {
        uv.U = cg.solve(b_u);
        // Check solve convergence immediately (before compute(A_v) overwrites info)
        if (cg.info() != Eigen::Success) {
            SparseMat A_pin_u = A_u;
            A_pin_u.coeffRef(pin_vertex_igm, pin_vertex_igm) += 1e4;
            Eigen::SimplicialLDLT<SparseMat> ldlt_u;
            ldlt_u.compute(A_pin_u);
            if (ldlt_u.info() == Eigen::Success)
                uv.U = ldlt_u.solve(b_u);
        }
    } else {
        // Fallback: SimplicialLDLT with soft pin to remove translation DOF.
        SparseMat A_pin_u = A_u;
        A_pin_u.coeffRef(pin_vertex_igm, pin_vertex_igm) += 1e4;
        Eigen::SimplicialLDLT<SparseMat> ldlt_u;
        ldlt_u.compute(A_pin_u);
        if (ldlt_u.info() == Eigen::Success)
            uv.U = ldlt_u.solve(b_u);
        // If LDLT also fails, uv.U stays zero — Step 3/4 will produce
        // zero jumps and a flat parametrisation, which is at least safe
        // (no NaN/Inf) and triggers the global-snap no-op in Step 5.
    }

    cg.compute(A_v);   // A_v == A_u (B5 fix) — preconditioner already set;
                       // re-compute on same matrix is safe and negligible cost.
    if (cg.info() == Eigen::Success) {
        uv.V = cg.solve(b_v);
        if (cg.info() != Eigen::Success) {
            SparseMat A_pin_v = A_v;
            A_pin_v.coeffRef(pin_vertex_igm, pin_vertex_igm) += 1e4;
            Eigen::SimplicialLDLT<SparseMat> ldlt_v;
            ldlt_v.compute(A_pin_v);
            if (ldlt_v.info() == Eigen::Success)
                uv.V = ldlt_v.solve(b_v);
        }
    } else {
        SparseMat A_pin_v = A_v;
        A_pin_v.coeffRef(pin_vertex_igm, pin_vertex_igm) += 1e4;
        Eigen::SimplicialLDLT<SparseMat> ldlt_v;
        ldlt_v.compute(A_pin_v);
        if (ldlt_v.info() == Eigen::Success)
            uv.V = ldlt_v.solve(b_v);
    }

    const auto& seams = combing.seam_edges;
    int ns = (int)seams.size();

    if (ns == 0) {
        // No seams: global snap is safe (simply-connected surface)
        if (params.strict_global_snap) {
            for (int vi = 0; vi < nv; ++vi) {
                uv.U[vi] = std::round(uv.U[vi]);
                uv.V[vi] = std::round(uv.V[vi]);
            }
        }
        return uv;
    }

    // ----------------------------------------------------------------
    // Step 2: Compute raw period jumps at each seam edge
    // The continuous solve gives UV values that should differ by
    // near-integers across each seam.  We snap those to integers.
    // ----------------------------------------------------------------
    std::vector<double> raw_ju(ns), raw_jv(ns);
    std::vector<int>    int_ju(ns), int_jv(ns);

    for (int k = 0; k < ns; ++k) {
        int v0 = seams[k].first, v1 = seams[k].second;
        if (v0 < 0 || v0 >= nv || v1 < 0 || v1 >= nv) continue;

        raw_ju[k] = uv.U[v1] - uv.U[v0];
        raw_jv[k] = uv.V[v1] - uv.V[v0];

        // BUG FIX (Bug 6): IGM always rounds period jumps to the nearest
        // integer — no conditional branching needed.  The previous code had
        // two identical ternary branches, making the condition dead code.
        // The correct behaviour is unconditional rounding (Bommes 2013).
        int_ju[k] = (int)std::round(raw_ju[k]);
        int_jv[k] = (int)std::round(raw_jv[k]);
    }

    // ----------------------------------------------------------------
    // Step 3: Globally consistent rounding via Union-Find + offsets
    //
    // Process seam edges as a spanning tree.  For each tree edge we
    // enforce U[v1] - U[v0] = int_ju[k] exactly by propagating an
    // integer offset to all vertices in v1's component.
    // Co-tree edges are then checked for consistency.
    // ----------------------------------------------------------------
    UFOffset uf(nv);
    std::vector<bool> is_cotree(ns, false);

    for (int k = 0; k < ns; ++k) {
        int v0 = seams[k].first, v1 = seams[k].second;
        if (v0 < 0 || v0 >= nv || v1 < 0 || v1 >= nv) {
            is_cotree[k] = true;
            continue;
        }

        bool ok = uf.unite(v0, v1,
                           (double)int_ju[k],
                           (double)int_jv[k],
                           0.5);
        if (!ok) {
            // Contradiction: this is a co-tree edge
            is_cotree[k] = true;
        }
    }

    // Repair co-tree edges: adjust their jump to match the
    // existing spanning-tree constraints
    for (int k = 0; k < ns; ++k) {
        if (!is_cotree[k]) continue;
        int v0 = seams[k].first, v1 = seams[k].second;
        if (v0 < 0 || v0 >= nv || v1 < 0 || v1 >= nv) continue;

        double off0u, off0v, off1u, off1v;
        uf.get_offset(v0, off0u, off0v);
        uf.get_offset(v1, off1u, off1v);

        // Required jump: off1 - off0 (since U[root] + off = U[v])
        double required_u = off1u - off0u;
        double required_v = off1v - off0v;
        int_ju[k] = (int)std::round(required_u);
        int_jv[k] = (int)std::round(required_v);
    }

    // BUG FIX (P1 — param/igm v111): Update uv.period_jumps with the actual
    // IGM-computed integer UV period jumps (int_ju / int_jv) instead of the
    // combing rotation counts stored at the top of this function.
    //
    // Root cause: uv.period_jumps was populated at function entry from
    // combing.period_jumps, which stores 4-RoSy ROTATION COUNTS in [0,3]
    // packed as (du=0, dv=rotation_count).  This is a combing-field concept,
    // not a UV-space concept.  The downstream consumer build_period_jump_map()
    // and extract_quads_from_isolines_seam_aware() interpret the (Δu, Δv)
    // pairs as actual INTEGER UV DIFFERENCES across each seam edge:
    //
    //     U[v_hi] ≈ U[v_lo] + Δu
    //     V[v_hi] ≈ V[v_lo] + Δv
    //
    // For a seam where the combing rotation is k (a 4-RoSy rotation by k×90°),
    // the combing convention (du=0, dv=k) is NOT the same as the actual UV
    // difference enforced by the IGM penalty re-solve.  The constrained
    // re-solve (Step 4) enforces U[v1]-U[v0] = int_ju[k] and
    // V[v1]-V[v0] = int_jv[k], which the isoline tracer MUST know to cross
    // seams correctly.  Using (0, rotation_count) instead of (int_ju, int_jv)
    // causes the isoline tracer to apply wrong UV offsets at seam crossings,
    // producing T-junctions, misaligned quad patches, and broken topology
    // wherever the rotation-count and actual-UV-difference disagree.
    //
    // The fix replaces the initial rotation-count entries with the globally
    // consistent int_ju / int_jv computed in Steps 2–3 above.  For seam edges
    // that are out of bounds (guard skipped them in Step 2), the (0,0) entry
    // from the initial population is left unchanged — this is the conservative
    // correct behaviour (no UV shift = no isoline tracing adjustment).
    //
    // Postcondition: uv.seam_edges.size() == uv.period_jumps.size() (unchanged).
    {
        const int n_period = static_cast<int>(uv.period_jumps.size());
        for (int k = 0; k < ns && k < n_period; ++k) {
            int v0 = seams[k].first, v1 = seams[k].second;
            if (v0 < 0 || v0 >= nv || v1 < 0 || v1 >= nv) continue;
            // int_ju[k] / int_jv[k] hold the globally-consistent integer UV
            // differences after co-tree repair (Step 3).  These are the
            // correct period jumps for seam-aware isoline tracing.
            uv.period_jumps[k] = { int_ju[k], int_jv[k] };
        }
    }

    // ----------------------------------------------------------------
    // Step 4: Constrained re-solve
    //
    // We modify the Poisson system to enforce the chosen integer jumps
    // exactly.  For each seam edge (v0, v1):
    //   U[v1] - U[v0] = int_ju  →  add a penalty constraint
    //     (U[v1] - U[v0] - int_ju)² with a large weight
    //
    // This produces a new system that honours the integer transitions.
    // ----------------------------------------------------------------
    const double PENALTY = 1e6;

    // Clone the Poisson matrices and add penalty terms
    SparseMat Au_c = A_u;
    SparseMat Av_c = A_v;
    VecX bu_c = b_u;
    VecX bv_c = b_v;

    for (int k = 0; k < ns; ++k) {
        int v0 = seams[k].first, v1 = seams[k].second;
        if (v0 < 0 || v0 >= nv || v1 < 0 || v1 >= nv) continue;

        // Penalty for U: w*(U[v0] - U[v1] + int_ju)^2
        // Gradient wrt U[v0]: 2w*(U[v0] - U[v1] + int_ju)
        // → A[v0,v0] += w, A[v0,v1] -= w, b[v0] -= w*int_ju
        //   A[v1,v1] += w, A[v1,v0] -= w, b[v1] += w*int_ju

        Au_c.coeffRef(v0, v0) += PENALTY;
        Au_c.coeffRef(v0, v1) -= PENALTY;
        Au_c.coeffRef(v1, v1) += PENALTY;
        Au_c.coeffRef(v1, v0) -= PENALTY;
        bu_c[v0] -= PENALTY * (double)int_ju[k];
        bu_c[v1] += PENALTY * (double)int_ju[k];

        Av_c.coeffRef(v0, v0) += PENALTY;
        Av_c.coeffRef(v0, v1) -= PENALTY;
        Av_c.coeffRef(v1, v1) += PENALTY;
        Av_c.coeffRef(v1, v0) -= PENALTY;
        bv_c[v0] -= PENALTY * (double)int_jv[k];
        bv_c[v1] += PENALTY * (double)int_jv[k];
    }

    // Re-solve with penalty-augmented systems
    Eigen::ConjugateGradient<SparseMat,
                             Eigen::Lower|Eigen::Upper> cg2;
    cg2.setMaxIterations(params.max_cg_iterations * 2);
    cg2.setTolerance(params.cg_tolerance * 0.1);

    // BUG FIX (v63 — param/Bug B): Use `== Eigen::Success` instead of
    // `!= Eigen::NumericalIssue` for the constrained re-solve.
    //
    // Eigen's ConjugateGradient::info() returns one of three values:
    //   Eigen::Success          — factorisation succeeded AND solve converged
    //   Eigen::NumericalIssue   — factorisation/preconditioner failed
    //   Eigen::NoConvergence    — factorisation ok but solve did not converge
    //
    // `!= NumericalIssue` accepts BOTH Success AND NoConvergence.  Accepting
    // NoConvergence means silently accepting a solver result that failed to
    // converge — producing inaccurate UV values that look plausible but are
    // wrong, corrupting every downstream stage (integer snap, iso-line tracing,
    // quad extraction).  The penalty-augmented system (Step 4) is stiffer than
    // the original Poisson system, so NoConvergence is MORE likely here than in
    // the initial solve at lines 209 and 213, which correctly use == Success.
    //
    // The fix is consistent with the identical correction already applied to
    // the initial CG solve (lines 209/213) and to miq.cpp::compute_parametrization
    // (BUG FIX Bug 5).  On failure, uv.U/uv.V retain their Step-1 values, which
    // are a valid (non-integer-rounded) parametrisation — degrading gracefully to
    // Poisson-only quality rather than returning garbage.
    //
    // BUG FIX (Bug 3-step4 — param/igm): After cg2.solve(bu_c), cg2.info()
    // is updated to Success or NoConvergence for the U solve.  But the very
    // next statement was cg2.compute(Av_c), which OVERWRITES cg2.info() before
    // the U-solve result could be inspected.  If the stiffer U solve returned
    // NoConvergence, the partially-converged uv.U was silently stored and
    // subsequently used to compute period jumps in Step 2 — cascading garbage
    // into integer rounding and all downstream stages.
    //
    // Fix: snapshot Step-1 U/V, check cg2.info() immediately after each
    // solve() call (before the next compute() resets it), and revert to the
    // Step-1 result on non-convergence so the output degrades gracefully.
    const Eigen::VectorXd u_step1 = uv.U;   // Step-1 fallback snapshot
    const Eigen::VectorXd v_step1 = uv.V;

    cg2.compute(Au_c);
    if (cg2.info() == Eigen::Success) {
        uv.U = cg2.solve(bu_c);
        // Check solve convergence immediately — before compute(Av_c) overwrites info.
        if (cg2.info() != Eigen::Success)
            uv.U = u_step1;   // revert: Step-1 Poisson-only result is safer than garbage
    }

    cg2.compute(Av_c);
    if (cg2.info() == Eigen::Success) {
        uv.V = cg2.solve(bv_c);
        // Check solve convergence immediately.
        if (cg2.info() != Eigen::Success)
            uv.V = v_step1;   // revert to Step-1 result
    }

    // ----------------------------------------------------------------
    // Step 5: Strict global integer snap
    //
    // After the constrained re-solve, all vertices should be very close
    // to integer grid positions.  We snap them to exact integers with
    // a tolerance check: only snap if the fractional part is small.
    // ----------------------------------------------------------------
    if (params.strict_global_snap) {
        // Collect seam vertex set for separate treatment
        std::unordered_set<int> seam_verts;
        seam_verts.reserve(ns * 2);
        for (auto& [v0, v1] : seams) {
            seam_verts.insert(v0);
            seam_verts.insert(v1);
        }

        // Interior (non-seam) vertices: snap freely
        for (int vi = 0; vi < nv; ++vi) {
            if (seam_verts.count(vi)) continue;
            double fu = std::abs(uv.U[vi] - std::round(uv.U[vi]));
            double fv = std::abs(uv.V[vi] - std::round(uv.V[vi]));
            if (fu <= params.global_snap_tol) uv.U[vi] = std::round(uv.U[vi]);
            if (fv <= params.global_snap_tol) uv.V[vi] = std::round(uv.V[vi]);
        }

        // Seam vertices: snap with a tighter tolerance to preserve jumps
        const double seam_snap_tol = params.global_snap_tol * 0.5;
        for (int vi : seam_verts) {
            if (vi < 0 || vi >= nv) continue;
            double fu = std::abs(uv.U[vi] - std::round(uv.U[vi]));
            double fv = std::abs(uv.V[vi] - std::round(uv.V[vi]));
            if (fu <= seam_snap_tol) uv.U[vi] = std::round(uv.U[vi]);
            if (fv <= seam_snap_tol) uv.V[vi] = std::round(uv.V[vi]);
        }
    }

    return uv;
}

} // namespace qf
