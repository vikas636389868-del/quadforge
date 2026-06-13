/**
 * curvature.cpp — Discrete principal curvature (Rusinkiewicz 2004).
 *
 * Per-edge curvature estimates are assembled into a per-vertex quadratic
 * form, then eigen-decomposed to get principal directions and magnitudes.
 */

#include "../../include/quadforge/field/curvature.h"
#include "../../include/quadforge/accel/omp_utils.h"
// INTEGRATION FIX: cache_utils.h was not included — weights[] is a hot
// per-vertex array accumulated across all edges; 64-byte alignment lets
// the parallel accumulation region use aligned SIMD stores.
#include "../../include/quadforge/accel/cache_utils.h"

#include <cmath>
#include <algorithm>
#include <Eigen/Eigenvalues>

namespace qf {

CurvatureData compute_curvature(const HalfEdgeMesh& mesh, int num_threads) {
    int nv = mesh.num_vertices();
    int nf = mesh.num_faces();

    // BUG FIX (v25): previously num_threads was accepted but never used.
    // Now passed to parallel regions through resolve_thread_count.
    int nt = resolve_thread_count(num_threads);

    CurvatureData data;
    data.kappa1.setZero(nv);
    data.kappa2.setZero(nv);
    data.dir1.setZero(nv, 3);
    data.dir2.setZero(nv, 3);
    data.mean_curv.setZero(nv);
    data.gauss_curv.setZero(nv);
    data.max_abs_curv.setZero(nv);

    // Per-vertex 3×3 symmetric curvature tensor (in world space)
    std::vector<Mat3> tensors(nv, Mat3::Zero());
    // INTEGRATION FIX: was std::vector<double> — upgraded to aligned_vector
    // so per-vertex weight accumulation is 64-byte aligned for SIMD stores.
    qf::aligned_vector<double> weights(nv, 0.0);

    // Precompute face normals
    // BUG FIX (v59): For zero-area (degenerate) faces, mesh.face_normal() returns
    // the zero vector.  Calling .normalized() on it produces NaN (0/0) in Eigen.
    // That NaN then propagates into every kappa_e computed along any edge that
    // borders the degenerate face, silently writing NaN into the curvature tensors
    // of both endpoint vertices — and from there into kappa1, kappa2, dir1, dir2,
    // etc. — corrupting the entire curvature output for those vertices.
    // Fix: skip the normalisation call on zero-magnitude normals and store a zero
    // sentinel instead.  The edge loop below already skips edges between faces
    // where fi < 0 || fj < 0, but a degenerate face still gets a valid face index.
    // We detect degenerate faces by checking the raw normal length, and store zero
    // for them; then in the edge loop we additionally skip any edge that touches a
    // degenerate face (norm < threshold) to prevent NaN from entering kappa_e.
    std::vector<Vec3> fnormals(nf);
    std::vector<bool> degenerate_face(nf, false);
    for (int fi = 0; fi < nf; ++fi) {
        Vec3 raw = mesh.face_normal(fi);
        double rlen = raw.norm();
        if (rlen > 1e-14) {
            fnormals[fi] = raw / rlen;
        } else {
            fnormals[fi] = Vec3::Zero(); // sentinel — edge loop will skip this face
            degenerate_face[fi] = true;
        }
    }

    // For each interior edge, compute normal curvature κ_e and contribute
    // to both endpoint vertices' tensors
    int nhe = mesh.num_half_edges();
    for (int he = 0; he < nhe; ++he) {
        int tw = mesh.half_edge(he).twin;
        if (tw < 0) continue; // boundary edge
        if (tw < he) continue; // process once

        int fi = mesh.half_edge(he).face;
        int fj = mesh.half_edge(tw).face;
        if (fi < 0 || fj < 0) continue;
        // BUG FIX (v59) continued: also skip edges that touch a degenerate face
        // so that the zero sentinel normal never enters the kappa_e formula.
        if (degenerate_face[fi] || degenerate_face[fj]) continue;

        // Edge vertices
        int from = mesh.half_edge(mesh.half_edge(he).prev).vertex;
        int to   = mesh.half_edge(he).vertex;

        Vec3 edge = mesh.vertex_pos(to) - mesh.vertex_pos(from);
        double edge_len = edge.norm();
        if (edge_len < 1e-15) continue;
        Vec3 e_dir = edge / edge_len;

        // Normal curvature: κ_e = 2*(n1 - n2)·e / |e|²
        double kappa_e = 2.0 * (fnormals[fi] - fnormals[fj]).dot(e_dir) / edge_len;

        // Weight: edge length (area weighting)
        double w = edge_len;

        // Contribution to outer product: T += w * kappa_e * (e⊗e)
        Mat3 outer = e_dir * e_dir.transpose();

        // Contribute to both endpoint vertices
        for (int vi : {from, to}) {
            tensors[vi] += w * kappa_e * outer;
            weights[vi] += w;
        }
    }

    // Eigen-decompose each vertex tensor (embarrassingly parallel — each vertex independent)
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64) num_threads(nt)
#endif
    for (int vi = 0; vi < nv; ++vi) {
        if (weights[vi] < 1e-15) continue;

        Mat3 T = tensors[vi] / weights[vi];

        Eigen::SelfAdjointEigenSolver<Mat3> solver(T);
        Vec3 evals = solver.eigenvalues();
        Eigen::Matrix3d evecs = solver.eigenvectors();

        // Sort by magnitude descending
        int idx[3] = {0, 1, 2};
        std::sort(idx, idx+3, [&](int a, int b){
            return std::abs(evals[a]) > std::abs(evals[b]);
        });

        data.kappa1[vi] = evals[idx[0]];
        data.kappa2[vi] = evals[idx[1]];

        // BUG FIX (v25): project curvature directions onto the vertex tangent
        // plane.  Raw eigenvectors of the shape tensor may have a small normal
        // component due to numerical error; leaving them un-projected causes
        // the cross-field seeding to produce slightly off-surface directions
        // that accumulate into visible field artefacts on low-poly meshes.
        // We approximate the vertex normal as the area-weighted average of
        // adjacent face normals.
        Vec3 vn = Vec3::Zero();
        mesh.vertex_ring(vi, [&](int he) {
            int fi = mesh.half_edge(he).face;
            if (fi >= 0) {
                auto [v0, v1, v2] = mesh.face_vertices(fi);
                Vec3 ab = mesh.vertex_pos(v1) - mesh.vertex_pos(v0);
                Vec3 ac = mesh.vertex_pos(v2) - mesh.vertex_pos(v0);
                Vec3 fn = ab.cross(ac);          // magnitude = 2 * area → area weight
                vn += fn;
            }
        });
        double vn_len = vn.norm();

        auto project_to_tangent = [&](const Vec3& dir) -> Vec3 {
            if (vn_len < 1e-12) return dir.normalized();
            Vec3 n  = vn / vn_len;
            Vec3 t  = dir - dir.dot(n) * n;
            double tlen = t.norm();
            return (tlen > 1e-12) ? (t / tlen) : n.unitOrthogonal();
        };

        Vec3 d1 = project_to_tangent(evecs.col(idx[0]));
        Vec3 d2 = project_to_tangent(evecs.col(idx[1]));

        data.dir1.row(vi) = d1.transpose();
        data.dir2.row(vi) = d2.transpose();

        data.mean_curv[vi]    = (evals[idx[0]] + evals[idx[1]]) * 0.5;
        data.gauss_curv[vi]   = evals[idx[0]] * evals[idx[1]];
        data.max_abs_curv[vi] = std::max(std::abs(evals[idx[0]]),
                                          std::abs(evals[idx[1]]));
    }

    return data;
}

} // namespace qf
