/**
 * singularity.cpp — Singularity detection and optimisation for 4-RoSy fields.
 *
 * FIX (v55): O(n²) → O(n) symmetry enforcement.
 *   The enforce_symmetry branch in optimise_singularities() previously used
 *   a nested O(n²) double-loop over all face pairs to find mirror matches.
 *   On a 100K-face mesh this is ~10^10 iterations — completely unusable.
 *   Replaced with a two-pass grid-hash approach: bucket faces by their (y, z)
 *   centroid coordinates (after quantising to a cell size), then for each face
 *   with cx > 0 look up only the single bucket for (-cx, cy, cz).
 *   Total cost: O(n) expected with the hash map.
 *
 * FIX (v55): Circular mean in blend_ring().
 *   The old code averaged face_frames with a plain arithmetic mean, which is
 *   incorrect when angles span the ±π/4 (4-RoSy) wrap boundary.  Replaced
 *   with the circular mean: avg = atan2(Σ sin(4θ), Σ cos(4θ)) / 4, which
 *   always lands on the geodesic midpoint of the angle set modulo π/2.
 *
 * FIX (v94): Added flat-array overload of detect_singularities() so that
 *   test_crossfield.cpp (and any tool using the low-level connection Laplacian
 *   API) can detect singularities without building a full HalfEdgeMesh.
 */

#include "../../include/quadforge/field/singularity.h"
#include "../../include/quadforge/field/cross_field.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <cstdint>
#include <vector>   // explicit — required by flat-array detect_singularities overload
#include <array>    // std::array used in face-vertex helper

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qf {

std::vector<SingularityInfo> detect_singularities(
    const HalfEdgeMesh& mesh,
    const CrossField&   field)
{
    std::vector<SingularityInfo> result;
    int nv = mesh.num_vertices();

    // Helper: build a stable per-face tangent frame from the face's v0→v1 edge.
    // Identical logic to the fixed parallel_transport_angle in cross_field.cpp.
    auto make_stable_frame = [&](int fi, Vec3& e1, Vec3& e2) {
        auto [v0, v1, v2] = mesh.face_vertices(fi);
        // BUG FIX (v61): Guard against degenerate (zero-area) faces.
        // face_normal() returns zero for zero-area triangles; .normalized() on
        // zero produces NaN (0/0 in Eigen).  NaN then flows into the projection,
        // makes proj.norm() NaN, and the (len > 1e-12) fallback is silently
        // bypassed (NaN comparison is always false), propagating NaN into e1, e2,
        // phi_i/phi_j, and the holonomy sum — producing false ±1/4 singularity
        // detections on otherwise clean regions of the mesh.
        // Fix: measure the raw normal length; fall back to canonical Z-axis if
        // degenerate (same strategy as compute_face_frames in cross_field.cpp
        // lines 53-58).  Any face frame built from a degenerate face is arbitrary
        // but finite, which is far safer than NaN propagation.
        Vec3 raw_n = mesh.face_normal(fi);
        double n_len = raw_n.norm();
        Vec3 n = (n_len > 1e-14) ? (raw_n / n_len) : Vec3(0, 0, 1);
        // BUG FIX (v60): Pass the raw edge (not pre-normalized).  If v0 == v1 in
        // position, .normalized() returns NaN, which makes proj.norm() NaN too,
        // and the guard (len > 1e-12) silently fails (NaN comparisons are false),
        // propagating NaN into e1/e2 and then into the holonomy sum.
        Vec3 ref = mesh.vertex_pos(v1) - mesh.vertex_pos(v0);
        Vec3 proj = ref - ref.dot(n) * n;
        double len = proj.norm();
        e1 = (len > 1e-12) ? proj / len : n.unitOrthogonal();
        e2 = n.cross(e1).normalized();
    };

    for (int vi = 0; vi < nv; ++vi) {
        if (mesh.is_boundary_vertex(vi)) continue;

        // BUG FIX (v25): previous code computed holonomy as Σ(aj - ai) without
        // subtracting the parallel-transport angle φ_ij for each edge.  On
        // curved surfaces the raw frame differences include the surface bending
        // contribution, which causes false singularity detections on smooth
        // curved regions and missed detections near actual singularities.
        //
        // Correct holonomy for a 4-RoSy field:
        //   holonomy = Σ_edges [ (θ_fj - θ_fi) - φ_ij ]  mod π/2
        // where φ_ij is the geometric parallel-transport angle between the two
        // adjacent face frames measured along the shared edge.
        double holonomy = 0.0;
        int ring_count  = 0;

        mesh.vertex_ring(vi, [&](int he) {
            int tw = mesh.half_edge(he).twin;
            if (tw < 0) return;
            int fi = mesh.half_edge(he).face;
            int fj = mesh.half_edge(tw).face;
            if (fi < 0 || fj < 0) return;

            double ai = field.face_frames[fi];
            double aj = field.face_frames[fj];

            // Geometric parallel-transport angle along the shared edge
            Vec3 e1_i, e2_i, e1_j, e2_j;
            make_stable_frame(fi, e1_i, e2_i);
            make_stable_frame(fj, e1_j, e2_j);

            int from = mesh.half_edge(mesh.half_edge(he).prev).vertex;
            int to   = mesh.half_edge(he).vertex;
            // BUG FIX (v60): Guard the shared-edge normalisation.  A zero-length
            // edge (coincident endpoints) makes .normalized() return NaN, which
            // then corrupts phi_i, phi_j, transport, diff, and the holonomy sum —
            // producing false ±1/4 singularity detections on clean curved regions.
            // Skip: a zero-length edge contributes zero to the holonomy integral.
            Vec3 raw_e = mesh.vertex_pos(to) - mesh.vertex_pos(from);
            double elen = raw_e.norm();
            if (elen < 1e-12) return;
            Vec3 edge = raw_e / elen;

            double phi_i = std::atan2(edge.dot(e2_i), edge.dot(e1_i));
            double phi_j = std::atan2(edge.dot(e2_j), edge.dot(e1_j));
            double transport = phi_j - phi_i;

            // Net field rotation corrected for geometric bending
            double diff = (aj - ai) - transport;
            // Wrap to [-π/4, π/4) for 4-RoSy
            diff -= std::round(diff / (M_PI / 2.0)) * (M_PI / 2.0);
            holonomy += diff;
            ++ring_count;
        });

        if (ring_count == 0) continue;

        // Singularity index = total holonomy / (2π), rounded to nearest ±0.25
        double idx     = holonomy / (2.0 * M_PI);
        double rounded = std::round(idx * 4.0) / 4.0;
        if (std::abs(rounded) >= 0.2) {
            SingularityInfo s;
            s.vertex_index = vi;
            s.index        = rounded;
            s.position     = mesh.vertex_pos(vi);
            result.push_back(s);
        }
    }
    return result;
}

CrossField optimise_singularities(
    const HalfEdgeMesh&             mesh,
    const CrossField&               field,
    const std::vector<int>&         corner_vertices,
    bool                            enforce_symmetry)
{
    CrossField result = field;

    // -----------------------------------------------------------------------
    // Strategy 1: Cancel adjacent +1/4 / -1/4 singularity pairs.
    //
    // Two singularities with opposite index that are adjacent (sharing a face
    // or connected by a short path) can be cancelled by smoothing the field
    // between them. We identify such pairs greedily (nearest-pair first) and
    // average the field angles in the path between them.
    // -----------------------------------------------------------------------

    // Build singularity list from current field
    auto sings = detect_singularities(mesh, result);
    if (sings.size() < 2) return result;

    // Separate into positive and negative index singularities
    std::vector<int> pos_sings, neg_sings;
    for (auto& s : sings) {
        if (s.index > 0) pos_sings.push_back(s.vertex_index);
        else             neg_sings.push_back(s.vertex_index);
    }

    // Mark cancelled vertices to avoid double-processing
    std::vector<bool> cancelled(mesh.num_vertices(), false);

    // For each positive singularity find the nearest negative one;
    // if they are "close" (within 3 ring hops), cancel them by
    // blending the face_frames in the ring between them.
    for (int pv : pos_sings) {
        if (cancelled[pv]) continue;

        // Find closest uncancelled negative singularity by geodesic distance
        // (approximated here as Euclidean distance — good enough for local pairs)
        int   best_nv   = -1;
        double best_dist = std::numeric_limits<double>::max();
        Vec3   ppos     = mesh.vertex_pos(pv);

        for (int nv : neg_sings) {
            if (cancelled[nv]) continue;
            double d = (mesh.vertex_pos(nv) - ppos).squaredNorm();
            if (d < best_dist) { best_dist = d; best_nv = nv; }
        }
        if (best_nv < 0) continue;

        // Compute a rough local edge-length estimate from the vertex ring
        double local_len = 0.0;
        int ring_count = 0;
        mesh.vertex_ring(pv, [&](int he) {
            int from = mesh.half_edge(mesh.half_edge(he).prev).vertex;
            int to   = mesh.half_edge(he).vertex;
            local_len += (mesh.vertex_pos(to) - mesh.vertex_pos(from)).norm();
            ++ring_count;
        });
        if (ring_count > 0) local_len /= ring_count;
        else                local_len  = 1e-3;

        // Only cancel if the pair is within ~5 edge lengths of each other
        // (otherwise cancellation would distort the field too much)
        if (best_dist > (5.0 * local_len) * (5.0 * local_len)) continue;

        // Smooth the face_frames in both vertex rings to blend the transition.
        // This is a lightweight "field smoothing between singularities" pass:
        // average the face_frames of faces adjacent to both vertices.
        // BUG FIX (v57): The lambda parameter `target_blend` was declared but
        // immediately suppressed with /*…*/ and then hardcoded to 0.3/0.7 inside
        // the body.  This means the blend ratio was always 30% regardless of what
        // the caller passed.  Since the callsites use 0.3 the output happened to
        // be correct, but the dead parameter made the interface misleading and
        // prevented future tuning.  Fixed: the suppression marker is removed and
        // the body now uses `target_blend` / `(1.0 - target_blend)` throughout.
        auto blend_ring = [&](int vi, double target_blend) {
            // FIX (v55): Use circular mean for 4-RoSy angles.
            // A plain arithmetic mean of angles is incorrect when values
            // straddle the ±π/4 wrap boundary (e.g. averaging 0.49*π/2 and
            // -0.49*π/2 gives ~0 rather than the correct wrap-around mean).
            // The circular mean works in the 4× angle space: accumulate
            // sin(4θ) and cos(4θ), then back-divide by 4.
            double sin_sum = 0.0, cos_sum = 0.0;
            int count = 0;
            mesh.vertex_ring(vi, [&](int he) {
                int fi = mesh.half_edge(he).face;
                if (fi >= 0) {
                    double t4 = 4.0 * result.face_frames[fi];
                    sin_sum += std::sin(t4);
                    cos_sum += std::cos(t4);
                    ++count;
                }
            });
            if (count == 0) return;
            double avg = (std::abs(sin_sum) < 1e-15 && std::abs(cos_sum) < 1e-15)
                         ? 0.0
                         : std::atan2(sin_sum, cos_sum) * 0.25;
            mesh.vertex_ring(vi, [&](int he) {
                int fi = mesh.half_edge(he).face;
                if (fi >= 0) {
                    // BUG FIX (v60): Linear angle interpolation
                    //   face_frames = old*(1-t) + avg*t
                    // is incorrect when old and avg straddle the ±π/4 (4-RoSy)
                    // wrap boundary.  Example: old = π/4−ε, avg = −π/4+ε.
                    // The linear blend gives ~0, but the geodesic midpoint on the
                    // 4-RoSy unit circle is the boundary itself (±π/4), not zero.
                    // Fix: blend in the complex exp(4i·θ) representation, then
                    // recover the angle with arg()/4.  This is the same strategy
                    // the v55 circular-mean fix uses for the accumulation step.
                    std::complex<double> u_curr = result.face_field[fi];
                    std::complex<double> u_avg  =
                        std::exp(std::complex<double>(0, 4.0 * avg));
                    std::complex<double> blended =
                        u_curr * (1.0 - target_blend) + u_avg * target_blend;
                    double bm = std::abs(blended);
                    if (bm > 1e-15) blended /= bm; else blended = u_avg;
                    result.face_field[fi]  = blended;
                    result.face_frames[fi] = std::arg(blended) * 0.25;
                }
            });
        };

        blend_ring(pv, 0.3);
        blend_ring(best_nv, 0.3);

        cancelled[pv]     = true;
        cancelled[best_nv] = true;
    }

    // -----------------------------------------------------------------------
    // Strategy 2: Relocate remaining singularities to corner vertices.
    //
    // If a singularity is near a feature corner, snap its associated irregular
    // vertex behavior to that corner by blending the surrounding faces toward
    // the corner's curvature direction. This produces cleaner layouts because
    // valence-3/5 vertices at corners are expected and visually invisible.
    // -----------------------------------------------------------------------

    if (!corner_vertices.empty()) {
        // Re-detect singularities after cancellation
        auto sings2 = detect_singularities(mesh, result);
        std::vector<bool> snap_done(mesh.num_vertices(), false);

        for (auto& s : sings2) {
            if (cancelled[s.vertex_index]) continue;

            // Find the nearest corner vertex
            int   best_cv   = -1;
            double best_dist = std::numeric_limits<double>::max();
            Vec3   spos     = mesh.vertex_pos(s.vertex_index);

            for (int cv : corner_vertices) {
                double d = (mesh.vertex_pos(cv) - spos).squaredNorm();
                if (d < best_dist) { best_dist = d; best_cv = cv; }
            }
            if (best_cv < 0 || snap_done[best_cv]) continue;

            // Compute local edge length for the snap-distance threshold
            double local_len2 = 0.0;
            int rc2 = 0;
            mesh.vertex_ring(s.vertex_index, [&](int he) {
                int from = mesh.half_edge(mesh.half_edge(he).prev).vertex;
                int to   = mesh.half_edge(he).vertex;
                local_len2 += (mesh.vertex_pos(to) - mesh.vertex_pos(from)).norm();
                ++rc2;
            });
            if (rc2 > 0) local_len2 /= rc2;
            else         local_len2  = 1e-3;

            // Snap if within 3 edge lengths
            if (best_dist > (3.0 * local_len2) * (3.0 * local_len2)) continue;

            // BUG FIX (v56): Use circular mean (same as blend_ring() fixed in v55).
            // Plain arithmetic mean is wrong when face_frames straddle the
            // ±π/4 wrap boundary of the 4-RoSy representation.
            // Circular mean: accumulate sin(4θ) and cos(4θ), then atan2/4.
            double cs_sin = 0.0, cs_cos = 0.0; int cc = 0;
            mesh.vertex_ring(best_cv, [&](int he) {
                int fi = mesh.half_edge(he).face;
                if (fi >= 0) {
                    double t4 = 4.0 * result.face_frames[fi];
                    cs_sin += std::sin(t4);
                    cs_cos += std::cos(t4);
                    ++cc;
                }
            });
            if (cc == 0) continue;
            double cavg = (std::abs(cs_sin) < 1e-15 && std::abs(cs_cos) < 1e-15)
                          ? 0.0
                          : std::atan2(cs_sin, cs_cos) * 0.25;

            double ss_sin = 0.0, ss_cos = 0.0; int sc = 0;
            mesh.vertex_ring(s.vertex_index, [&](int he) {
                int fi = mesh.half_edge(he).face;
                if (fi >= 0) {
                    double t4 = 4.0 * result.face_frames[fi];
                    ss_sin += std::sin(t4);
                    ss_cos += std::cos(t4);
                    ++sc;
                }
            });
            if (sc == 0) continue;
            double savg = (std::abs(ss_sin) < 1e-15 && std::abs(ss_cos) < 1e-15)
                          ? 0.0
                          : std::atan2(ss_sin, ss_cos) * 0.25;

            // BUG FIX (v60): Pull corner ring toward singularity average (soft relocation).
            // Linear angle interpolation (cavg * 0.6 + savg * 0.4) is wrong when
            // the two circular-mean angles straddle the ±π/4 4-RoSy wrap boundary —
            // the arithmetic mean of e.g. π/4−ε and −π/4+ε is ~0, not the correct
            // geodesic midpoint.  Blend in complex (exp(4i·θ)) space instead.
            mesh.vertex_ring(best_cv, [&](int he) {
                int fi = mesh.half_edge(he).face;
                if (fi >= 0) {
                    std::complex<double> u_cavg =
                        std::exp(std::complex<double>(0, 4.0 * cavg));
                    std::complex<double> u_savg =
                        std::exp(std::complex<double>(0, 4.0 * savg));
                    std::complex<double> blended = u_cavg * 0.6 + u_savg * 0.4;
                    double bm = std::abs(blended);
                    if (bm > 1e-15) blended /= bm; else blended = u_cavg;
                    result.face_field[fi]  = blended;
                    result.face_frames[fi] = std::arg(blended) * 0.25;
                }
            });

            snap_done[best_cv] = true;
        }
    }

    // -----------------------------------------------------------------------
    // FIX (v55): Symmetry enforcement — O(n²) → O(n) via grid hash.
    //
    // The previous implementation iterated all pairs (fi, fj) with fi < fj,
    // which is O(n²) in the face count.  On a 100 K-face mesh this is ~5×10⁹
    // comparisons — completely impractical.
    //
    // New approach (two passes, O(n) expected):
    //   Pass 1: For each face, compute its centroid and quantise (y, z) to a
    //           grid of cell size `tol`.  Insert into a hash map keyed by
    //           (iy, iz).  For each face with cx > tol, also record its
    //           negated-x bucket key (iy, iz) so we can look up its mirror.
    //   Pass 2: For each face fi with cx > tol, query the hash map at the
    //           bucket (iy, iz) — if exactly one face fj lives there whose
    //           cx ≈ -ci.x and cy ≈ ci.y and cz ≈ ci.z, they are a mirror
    //           pair and we symmetrise them.
    //
    // This reduces the cost to O(n) hash-map operations.
    // -----------------------------------------------------------------------
    if (enforce_symmetry) {
        int nf = mesh.num_faces();

        // Compute face centroids
        std::vector<Vec3> centroids(nf);
        for (int fi = 0; fi < nf; ++fi) {
            auto [v0, v1, v2] = mesh.face_vertices(fi);
            centroids[fi] = (mesh.vertex_pos(v0) +
                             mesh.vertex_pos(v1) +
                             mesh.vertex_pos(v2)) / 3.0;
        }

        // Estimate tolerance from average edge length (sample up to 2000 edges)
        double total_el = 0.0; int el_count = 0;
        int nhe = mesh.num_half_edges();
        for (int he = 0; he < nhe && el_count < 2000; ++he) {
            int tw = mesh.half_edge(he).twin;
            if (tw < 0 || tw < he) continue;
            int from = mesh.half_edge(mesh.half_edge(he).prev).vertex;
            int to   = mesh.half_edge(he).vertex;
            total_el += (mesh.vertex_pos(to) - mesh.vertex_pos(from)).norm();
            ++el_count;
        }
        double tol = (el_count > 0) ? (total_el / el_count * 0.5) : 0.01;
        if (tol < 1e-10) tol = 1e-10;

        // Grid-hash key: quantised (iy, iz) packed into int64
        // We use (int32 iy) << 32 | (int32 iz).
        // Quantise: iy = round(cy / tol), iz = round(cz / tol)
        auto make_key = [&](double cy, double cz) -> int64_t {
            auto iy = static_cast<int32_t>(std::round(cy / tol));
            auto iz = static_cast<int32_t>(std::round(cz / tol));
            return (static_cast<int64_t>(iy) << 32) |
                   static_cast<int64_t>(static_cast<uint32_t>(iz));
        };

        // Pass 1: bucket every face by its quantised (y, z) centroid
        // Store: key → list of face indices
        std::unordered_map<int64_t, std::vector<int>> yz_bucket;
        yz_bucket.reserve(nf);
        for (int fi = 0; fi < nf; ++fi) {
            int64_t key = make_key(centroids[fi][1], centroids[fi][2]);
            yz_bucket[key].push_back(fi);
        }

        std::vector<bool> processed(nf, false);

        // Pass 2: for each face on the +x side, look up its mirror bucket
        for (int fi = 0; fi < nf; ++fi) {
            if (processed[fi]) continue;
            const Vec3& ci = centroids[fi];
            if (ci[0] < tol) continue;  // skip on-plane and -x faces

            // Mirror bucket key: same (y, z), but we must find a face with
            // cx ≈ -ci.x in the same (iy, iz) bucket.
            int64_t key = make_key(ci[1], ci[2]);
            auto it = yz_bucket.find(key);
            if (it == yz_bucket.end()) continue;

            int best_fj = -1;
            double best_err = tol * 2.0;  // must be within tol
            for (int fj : it->second) {
                if (processed[fj] || fj == fi) continue;
                const Vec3& cj = centroids[fj];
                // Mirror condition: cx mirror, cy/cz match within tol
                double err = std::abs(ci[0] + cj[0]) +
                             std::abs(ci[1] - cj[1]) +
                             std::abs(ci[2] - cj[2]);
                if (err < best_err) { best_err = err; best_fj = fj; }
            }
            if (best_fj < 0) continue;

            // Symmetrise the field frames.
            // For X-mirror symmetry, the reflected field satisfies
            //   θ_j = π/2 - θ_i   (modulo π/2)
            // We enforce this with a 30% soft blend toward the symmetric
            // midpoint.
            //
            // BUG FIX (v56): The previous code computed
            //   avg_blend = avg * 0.3  and then  face_frames[fi] += avg_blend
            // which ADDS 30% of the midpoint angle VALUE to the current frame
            // rather than interpolating 30% of the way toward the midpoint.
            // For example, if ai=0.1 and avg=0.685, the old code produced
            //   fi = 0.1 + 0.206 = 0.306  (wrong — not moving toward 0.685)
            // while the correct lerp gives
            //   fi = 0.1*0.7 + 0.685*0.3 = 0.276  (actually approaching avg).
            // The fix is a standard linear interpolation: (1-α)*current + α*target.
            double ai = result.face_frames[fi];
            double aj = result.face_frames[best_fj];

            // BUG FIX (v60): The previous code computed the symmetric midpoint
            // and the 30% blend using linear angle arithmetic:
            //   avg = (ai + (π/2 - aj)) / 2
            //   fi_new = ai*0.7 + avg*0.3
            // This is wrong when ai and (π/2−aj) straddle the ±π/4 4-RoSy wrap
            // boundary, producing a midpoint on the wrong side of the circle.
            //
            // Correct approach: work entirely in the complex exp(4i·θ) space.
            //   u_i = exp(4i·ai)
            //   The X-mirror symmetry target for u_i is the reflection of u_j:
            //     u_j_mirror = exp(4i·(π/2 − aj))
            //                = exp(2πi) · exp(−4i·aj)
            //                = conj(exp(4i·aj))  [since |u_j| = 1]
            //                = conj(u_j)
            //   Geodesic midpoint on the 4-RoSy circle:
            //     u_mid = normalise(u_i + conj(u_j))
            //   30% complex blend:
            //     u_i_new = normalise(u_i * 0.7 + u_mid * 0.3)
            //   The mirrored face j must satisfy u_j_new = conj(u_i_new).
            std::complex<double> u_i = std::exp(std::complex<double>(0, 4.0 * ai));
            std::complex<double> u_j = std::exp(std::complex<double>(0, 4.0 * aj));

            // Mirror of u_j in the X-symmetry sense
            std::complex<double> u_j_mirror = std::conj(u_j);

            // Geodesic midpoint in complex space
            std::complex<double> u_mid = u_i + u_j_mirror;
            double u_mid_m = std::abs(u_mid);
            if (u_mid_m > 1e-15) u_mid /= u_mid_m; else u_mid = u_i;

            // 30% blend of u_i toward the symmetric midpoint
            std::complex<double> blended_i = u_i * 0.7 + u_mid * 0.3;
            double bim = std::abs(blended_i);
            if (bim > 1e-15) blended_i /= bim; else blended_i = u_mid;

            // Face j is exactly the X-mirror of face i
            std::complex<double> blended_j = std::conj(blended_i);

            result.face_field[fi]      = blended_i;
            result.face_field[best_fj] = blended_j;
            result.face_frames[fi]     = std::arg(blended_i) * 0.25;
            result.face_frames[best_fj] = std::arg(blended_j) * 0.25;

            processed[fi] = processed[best_fj] = true;
        }
    }

    return result;
}

// -----------------------------------------------------------------------
// detect_singularities — flat-array overload (v94)
//
// Detects 4-RoSy field singularities directly from raw triangle arrays
// using discrete holonomy computation around each interior vertex.
//
// Algorithm (Knöppel 2013 §2, discrete version):
//   For each interior vertex v, traverse its ordered one-ring of faces.
//   For each consecutive pair of faces (fi → fj) in the ring:
//     1. Compute φ_{ij} = parallel transport angle from fi's frame to fj's
//        frame, measured along the shared directed edge vi → next_vi.
//        (Same formula as connection.cpp parallel_transport_angle, but
//         operating on flat e1/e2/normal arrays instead of HalfEdgeMesh.)
//     2. Compute Δθ = (face_angles[fj] - face_angles[fi]) - φ_{ij}
//     3. Wrap Δθ to [-π/4, π/4) using the 4-RoSy branch cut (round to
//        nearest multiple of π/2).
//     4. Accumulate holonomy H += Δθ.
//   After the full ring:
//     H ≈ +π/2 → +1/4 singularity (positive index, valence-3 like)
//     H ≈ -π/2 → -1/4 singularity (negative index, valence-5 like)
//     H ≈ ±π   → ±1/2 singularity (rare; e.g. umbilics on smooth surfaces)
//
// Boundary detection:
//   If, while traversing the ring via adjacent_face(), we hit a boundary
//   edge (adjacent_face returns -1 before completing the loop), the vertex
//   is on the mesh boundary and is skipped — boundary vertices can have
//   non-zero holonomy by definition and are not classified as singularities.
//
// Ring ordering:
//   Starting from an arbitrary face in adj_faces[v], we walk the ring by
//   repeatedly finding the face adjacent to the current face across the
//   edge (v, next_v), where next_v is the vertex AFTER v in the current
//   face's CCW winding order.  This gives a consistently CCW-ordered ring.
//   If the ring cannot be closed (incomplete or non-manifold adjacency),
//   the vertex is skipped.
// -----------------------------------------------------------------------

void detect_singularities(
    const float*   positions,
    const int32_t* tris,
    int32_t        nt,
    int32_t        nv,
    const float*   face_angles,
    const float*   e1,
    const float*   e2,
    const float*   /*face_normals*/,   // reserved for future degenerate-face guard
    const std::unordered_map<int64_t, std::vector<int32_t>>& edge_to_faces,
    std::vector<int32_t>& out_sing_vertices,
    std::vector<float>&   out_sing_indices)
{
    out_sing_vertices.clear();
    out_sing_indices.clear();

    if (nv <= 0 || nt <= 0 || !positions || !tris ||
        !face_angles || !e1 || !e2) return;

    const double PI_2 = M_PI / 2.0;
    const double PI_4 = M_PI / 4.0;
    const int64_t VMAX = 1LL << 24;  // matches constraints.cpp / connection.cpp

    // -----------------------------------------------------------------------
    // Build vertex → face adjacency (one vector per vertex)
    // -----------------------------------------------------------------------
    std::vector<std::vector<int32_t>> vert_faces(static_cast<size_t>(nv));
    for (int32_t fi = 0; fi < nt; ++fi)
        for (int k = 0; k < 3; ++k) {
            int32_t vi = tris[fi*3+k];
            if (vi >= 0 && vi < nv)
                vert_faces[static_cast<size_t>(vi)].push_back(fi);
        }

    // -----------------------------------------------------------------------
    // Helpers: ring traversal
    // -----------------------------------------------------------------------

    // Given face fi and vertex vi, return the vertex AFTER vi in CCW order
    // (i.e., tris[fi][pos_of_vi + 1] mod 3).  Returns -1 on error.
    auto next_vert_in_face = [&](int fi, int vi) -> int {
        for (int k = 0; k < 3; ++k)
            if (tris[fi*3+k] == vi)
                return tris[fi*3+(k+1)%3];
        return -1;
    };

    // Given the current face fi and vertex vi, return the next face in the
    // CCW ring around vi.  The shared edge is (vi, next_vert_in_face(fi, vi)).
    // Returns -1 if the edge is a boundary (no adjacent face other than fi).
    auto adjacent_face = [&](int fi, int vi) -> int {
        int vj = next_vert_in_face(fi, vi);
        if (vj < 0) return -1;
        int lo = std::min(vi, vj), hi = std::max(vi, vj);
        int64_t key = static_cast<int64_t>(lo) * VMAX
                    + static_cast<int64_t>(hi);
        auto it = edge_to_faces.find(key);
        if (it == edge_to_faces.end()) return -1;
        for (int32_t f : it->second)
            if (f != fi) return f;
        return -1;  // boundary edge
    };

    // -----------------------------------------------------------------------
    // Parallel transport angle φ_{fi → fj} for the directed edge (from_v → to_v).
    //
    // φ is defined as (angle of shared edge in fj's frame) minus
    // (angle of shared edge in fi's frame), wrapped to [-π/4, π/4).
    // Matches the computation in connection.cpp parallel_transport_angle().
    // -----------------------------------------------------------------------
    auto transport_angle = [&](int fi, int fj,
                                int from_v, int to_v) -> double {
        const float* pf = positions + from_v*3;
        const float* pt = positions + to_v*3;
        float ex = pt[0]-pf[0], ey = pt[1]-pf[1], ez = pt[2]-pf[2];
        float elen = std::sqrt(ex*ex + ey*ey + ez*ez);
        if (elen < 1e-12f) return 0.0;
        ex /= elen; ey /= elen; ez /= elen;

        const float* ei1 = e1 + fi*3;
        const float* ei2 = e2 + fi*3;
        const float* ej1 = e1 + fj*3;
        const float* ej2 = e2 + fj*3;

        double ai = std::atan2(
            static_cast<double>(ex*ei2[0] + ey*ei2[1] + ez*ei2[2]),
            static_cast<double>(ex*ei1[0] + ey*ei1[1] + ez*ei1[2]));
        double aj = std::atan2(
            static_cast<double>(ex*ej2[0] + ey*ej2[1] + ez*ej2[2]),
            static_cast<double>(ex*ej1[0] + ey*ej1[1] + ez*ej1[2]));

        double phi = aj - ai;
        // 4-RoSy wrap to [-π/4, π/4)
        while (phi >  PI_4) phi -= PI_2;
        while (phi < -PI_4) phi += PI_2;
        return phi;
    };

    // -----------------------------------------------------------------------
    // Main loop: one holonomy sum per vertex
    // -----------------------------------------------------------------------
    for (int32_t vi = 0; vi < nv; ++vi) {
        const auto& adj = vert_faces[static_cast<size_t>(vi)];
        if (adj.size() < 3) continue;   // need at least 3 faces for a valid ring

        // ----------------------------------------------------------------
        // Build an ordered CCW ring of faces around vi, starting from adj[0].
        // We walk via adjacent_face() until we close the loop or hit a boundary.
        // ----------------------------------------------------------------
        std::vector<int> ring;
        ring.reserve(adj.size() + 1);
        ring.push_back(adj[0]);

        bool is_boundary = false;
        const int max_steps = static_cast<int>(adj.size()) * 2 + 2;
        for (int step = 0; step < max_steps; ++step) {
            int cur  = ring.back();
            int next = adjacent_face(cur, vi);
            if (next < 0) {
                // Hit a boundary edge — vertex is on the mesh boundary
                is_boundary = true;
                break;
            }
            if (next == ring[0] && static_cast<int>(ring.size()) > 1) {
                // Closed the loop
                break;
            }
            // Guard against infinite loops on non-manifold meshes
            bool already_seen = false;
            for (int f : ring) if (f == next) { already_seen = true; break; }
            if (already_seen) break;
            ring.push_back(next);
        }

        // Skip boundary vertices and incomplete rings
        if (is_boundary) continue;
        if (static_cast<int>(ring.size()) != static_cast<int>(adj.size())) continue;

        // ----------------------------------------------------------------
        // Accumulate holonomy H = Σ_k [ (θ_{k+1} - θ_k) - φ_{k→k+1} ]
        // with each term wrapped to the nearest multiple of π/2 (4-RoSy
        // branch cut).
        // ----------------------------------------------------------------
        double holonomy = 0.0;
        const int n_ring = static_cast<int>(ring.size());
        bool ring_ok = true;

        for (int k = 0; k < n_ring; ++k) {
            int fi = ring[k];
            int fj = ring[(k+1) % n_ring];

            // Shared directed edge: vi → next vertex in fi's CCW order
            int edge_to = next_vert_in_face(fi, vi);
            if (edge_to < 0) { ring_ok = false; break; }

            double phi = transport_angle(fi, fj, vi, edge_to);

            // Field angle difference, rounded to nearest π/2 branch
            double dtheta = static_cast<double>(face_angles[fj])
                          - static_cast<double>(face_angles[fi])
                          - phi;
            // Round to nearest multiple of π/2 (4-RoSy symmetry group)
            dtheta -= PI_2 * std::round(dtheta / PI_2);

            holonomy += dtheta;
        }

        if (!ring_ok) continue;

        // ----------------------------------------------------------------
        // Classify: H / (π/2) ≈ integer index × 0.25
        // ----------------------------------------------------------------
        float idx_rounded = 0.f;
        double idx = holonomy / PI_2;  // should be close to an integer

        if      (std::abs(idx -  1.0) < 0.45) idx_rounded =  0.25f;  // +1/4
        else if (std::abs(idx +  1.0) < 0.45) idx_rounded = -0.25f;  // -1/4
        else if (std::abs(idx -  2.0) < 0.55) idx_rounded =  0.50f;  // +1/2 (umbilics)
        else if (std::abs(idx +  2.0) < 0.55) idx_rounded = -0.50f;  // -1/2

        if (idx_rounded != 0.f) {
            out_sing_vertices.push_back(vi);
            out_sing_indices.push_back(idx_rounded);
        }
    }
}

} // namespace qf
