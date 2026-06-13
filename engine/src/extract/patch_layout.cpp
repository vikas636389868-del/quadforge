/**
 * patch_layout.cpp — Motorcycle graph patch decomposition.
 *
 * Implements build_patch_layout(), build_motorcycle_graph(),
 * extract_quads_motorcycle(), patch_aspect_ratio(), and patch_quad_budget()
 * declared in include/quadforge/extract/patch_layout.h.
 *
 * ALGORITHM OVERVIEW
 * ------------------
 * The motorcycle graph algorithm (Eppstein 2008) decomposes a quad-mesh
 * surface into rectangular patches by tracing "riders" from singularities
 * and boundary corners along the cross-field direction.
 *
 * This implementation follows the Bommes et al. (2013) strategy of using
 * the UV gradient to determine rider direction rather than tracing directly
 * in 3D, which gives better alignment with the eventual quad grid.
 *
 * RIDER SEEDING
 * -------------
 * A "rider" is a directed path through the triangle mesh dual graph.  Each
 * rider:
 *   - starts at a PatchVertex (singularity or boundary corner)
 *   - carries a dominant direction: U-constant (travel in V direction) or
 *     V-constant (travel in U direction)
 *   - advances by crossing triangle edges that intersect the rider's iso-line
 *   - stops when it hits another rider's track, a mesh boundary, a different
 *     singularity, or exceeds the step limit
 *
 * Seeding rules for 4-RoSy fields:
 *   +¼ singularity (valence 3):  1 rider
 *   −¼ singularity (valence 5):  3 riders
 *   boundary corner (convex):    1 rider
 *   boundary corner (concave):   3 riders
 *
 * PATCH EXTRACTION
 * ----------------
 * After all riders halt, the surface is decomposed into patches by flood-
 * filling triangles between the motorcycle tracks.  A patch boundary is
 * any triangle edge that contains a point on a rider's track.
 *
 * For each patch:
 *   - Identify 4 corner PatchVertex nodes (or 3/5 for degenerate patches).
 *   - Run a Tutte parametrization (square boundary, uniform weights) to get
 *     a [0,1]^2 UV over the patch triangles.
 *   - Scale the UV by (target_quads_u × target_quads_v) and run the standard
 *     extract_quads_from_isolines() on the patch sub-mesh.
 *
 * PER-PATCH PARAMETRIZATION (Tutte)
 * ------------------------------------
 * For a 4-corner patch:
 *   - Map the 4 boundary track corners to (0,0), (1,0), (1,1), (0,1).
 *   - Map the boundary vertices between corners to uniformly spaced positions
 *     on the corresponding square edges.
 *   - Solve the Laplacian system (uniform weights) with the boundary pinned.
 *   - Interior vertices get UV from the Laplacian solution.
 *
 * For degenerate patches (n_corners ≠ 4):
 *   - Fall back to extract_quads_from_isolines_seam() using the global UV
 *     (same as extraction_method=0) on just the patch triangles.
 *
 * VERTEX MERGING
 * ---------------
 * After per-patch extraction, adjacent patches share boundary tracks.  The
 * two patches assign independent vertex positions to the same track vertices,
 * so a weld pass merges vertices within tolerance 1e-6.
 */

// FIX (v90): patch_layout.cpp defines build_motorcycle_graph() which is
// declared in motorcycle.h, but previously never included motorcycle.h.
// Without this include, the compiler cannot verify that the function
// signature here matches the declaration — any parameter-type drift would
// produce silent UB.  Adding the include enforces compile-time verification.
// Note: motorcycle.h already includes patch_layout.h, so include ordering
// is: motorcycle.h -> patch_layout.h (guard fires, skipped) -> ok.
#include "../../include/quadforge/extract/motorcycle.h"
#include "../../include/quadforge/extract/patch_layout.h"
#include "../../include/quadforge/extract/isolines.h"
#include "../../include/quadforge/extract/seam_utils.h"
#include "../../include/quadforge/extract/quad_mesh.h"
#include "../../include/quadforge/mesh/halfedge.h"

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <cassert>
#include <cstdint>
#include <array>
#include <vector>
#include <limits>

namespace qf {

// ==========================================================================
// Internal helpers
// ==========================================================================

namespace {

/** Directed-edge key identical to seam_utils / motorcycle.cpp. */
static inline int64_t dir_key(int a, int b) noexcept {
    return (static_cast<int64_t>(static_cast<uint32_t>(a)) << 32)
         |  static_cast<int64_t>(static_cast<uint32_t>(b));
}

/** Face area via cross product. */
static double tri_area(const Vec3& a, const Vec3& b, const Vec3& c) {
    Vec3 ab = b - a, ac = c - a;
    return 0.5 * ab.cross(ac).norm();
}

/** Dominant UV direction for a face: returns (du/ds, dv/ds) normalised. */
static std::pair<double,double> face_uv_gradient(
    int fi,
    const HalfEdgeMesh& mesh,
    const UVParam& uv)
{
    auto [v0, v1, v2] = mesh.face_vertices(fi);
    double u0 = uv.U[v0], u1 = uv.U[v1], u2 = uv.U[v2];
    double v_0= uv.V[v0], v_1= uv.V[v1], v_2= uv.V[v2];
    double du = std::max({std::abs(u1-u0), std::abs(u2-u1), std::abs(u2-u0)});
    double dv = std::max({std::abs(v_1-v_0), std::abs(v_2-v_1), std::abs(v_2-v_0)});
    return {du, dv};
}

// -----------------------------------------------------------------------
// Rider state for the motorcycle-graph tracing loop
// -----------------------------------------------------------------------

enum class RiderDir { U_CONST, V_CONST };  // riding along constant-U or constant-V

struct Rider {
    int    patch_vertex_idx;   // source PatchVertex
    int    current_face;       // current triangle being traversed
    int    src_v;              // last track vertex (mesh index)
    double iso_value;          // the integer U or V value being traced
    RiderDir dir;              // direction of travel
    std::vector<int> track;    // ordered mesh vertex indices on this track
    bool   halted = false;
};

// -----------------------------------------------------------------------
// Find the half-edge crossing of a given iso-line inside a triangle
// -----------------------------------------------------------------------
// Given face fi and iso_value on axis (U if dir==U_CONST, else V),
// returns the edge (va→vb) whose U (or V) bracket contains iso_value,
// together with the mesh vertex index of the opposite vertex (the vertex
// not on this edge, which we will enter from if we advance through the edge).
// Returns -1 for the edge vertex indices on failure.

struct EdgeCrossing {
    int va, vb;       // edge vertices (mesh indices)
    int twin_face;    // adjacent face across this edge (-1 = boundary)
    int twin_he;      // half-edge in twin_face
    double t;         // parametric position on edge [0,1]
    int track_vert;   // nearest mesh vertex (va or vb, whichever is closer to t)
};

static bool find_exit_edge(
    int fi,
    double iso_val,
    RiderDir dir,
    int entry_va, int entry_vb,    // edge we entered from (don't exit same edge)
    const HalfEdgeMesh& mesh,
    const UVParam& uv,
    const PeriodJumpMap& jmap,
    EdgeCrossing& out)
{
    auto [he0, he1, he2] = mesh.face_half_edges(fi);
    const int hes[3] = {he0, he1, he2};

    for (int hei : hes) {
        const HalfEdge& he = mesh.half_edge(hei);
        int prev_he = he.prev;
        if (prev_he < 0) continue;
        int va = mesh.half_edge(prev_he).vertex;
        int vb = he.vertex;

        // Skip the edge we entered from
        if ((va == entry_va && vb == entry_vb) ||
            (va == entry_vb && vb == entry_va)) continue;

        // Get consistent UV for this edge's endpoints
        double ua = (dir == RiderDir::U_CONST) ? uv.U[va] : uv.V[va];
        double ub = (dir == RiderDir::U_CONST) ? uv.U[vb] : uv.V[vb];

        // Apply period jump on the relevant axis when crossing a seam edge
        double ub_adj = ub;
        {
            auto it = jmap.find(seam_edge_key(va, vb));
            if (it != jmap.end()) {
                ub_adj += (dir == RiderDir::U_CONST)
                          ? static_cast<double>(it->second.du)
                          : static_cast<double>(it->second.dv);
            }
        }

        double lo = std::min(ua, ub_adj);
        double hi = std::max(ua, ub_adj);
        if (iso_val < lo - 1e-9 || iso_val > hi + 1e-9) continue;
        if (std::abs(hi - lo) < 1e-12) continue;

        double t = (iso_val - ua) / (ub_adj - ua);
        t = std::max(0.0, std::min(1.0, t));

        out.va = va;
        out.vb = vb;
        out.twin_face = -1;
        out.twin_he   = -1;
        out.t         = t;
        out.track_vert = (t < 0.5) ? va : vb;

        int twin_hei = he.twin;
        if (twin_hei >= 0) {
            out.twin_face = mesh.half_edge(twin_hei).face;
            out.twin_he   = twin_hei;
        }
        return true;
    }
    return false;
}

// -----------------------------------------------------------------------
// Solve Tutte parametrization for a patch sub-mesh
// -----------------------------------------------------------------------
// boundary_verts: ordered list of boundary mesh vertex indices for this patch
// boundary_uvs:   corresponding (U,V) in [0,1]^2 for each boundary vertex
// sub_verts:      all mesh vertex indices (boundary + interior)
// sub_faces:      triangles (as triplets of indices into sub_verts)
// Returns: U_patch, V_patch indexed by position in sub_verts

static bool solve_tutte(
    const std::vector<int>&                   sub_verts,
    const std::vector<std::array<int,3>>&      sub_faces_local,  // local indices
    const std::vector<int>&                   boundary_local,    // local boundary indices
    const std::vector<std::pair<double,double>>& boundary_uv,
    std::vector<double>& U_out,
    std::vector<double>& V_out)
{
    const int N = static_cast<int>(sub_verts.size());
    const int Nb = static_cast<int>(boundary_local.size());
    if (N == 0 || sub_faces_local.empty()) return false;

    U_out.assign(N, 0.0);
    V_out.assign(N, 0.0);

    // Pin boundary vertices immediately
    std::vector<bool> is_boundary(N, false);
    for (int k = 0; k < Nb; ++k) {
        int li = boundary_local[k];
        if (li < 0 || li >= N) continue;
        is_boundary[li] = true;
        U_out[li] = boundary_uv[k].first;
        V_out[li] = boundary_uv[k].second;
    }

    // Count interior vertices
    std::vector<int> interior;
    interior.reserve(N - Nb);
    for (int i = 0; i < N; ++i)
        if (!is_boundary[i]) interior.push_back(i);

    const int Ni = static_cast<int>(interior.size());
    if (Ni == 0) return true;  // only boundary — trivially solved

    // Build interior index map: local_idx → interior_idx
    std::vector<int> to_int(N, -1);
    for (int k = 0; k < Ni; ++k) to_int[interior[k]] = k;

    // Assemble uniform-weight Laplacian for interior DOFs.
    // For each triangle (i,j,k): edges i-j, j-k, k-i each contribute 1.
    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> triplets;
    triplets.reserve(sub_faces_local.size() * 6);

    Eigen::VectorXd rhs_u(Ni), rhs_v(Ni);
    rhs_u.setZero(); rhs_v.setZero();

    // degree[i] = number of adjacent vertices for interior vertex i
    std::vector<double> degree(Ni, 0.0);

    for (auto& tri : sub_faces_local) {
        const int abc[3] = {tri[0], tri[1], tri[2]};
        for (int e = 0; e < 3; ++e) {
            int li = abc[e];
            int lj = abc[(e+1)%3];
            int ii = to_int[li];
            int ij = to_int[lj];

            if (ii >= 0) {  // li is interior
                degree[ii] += 1.0;
                if (ij >= 0) {
                    // interior–interior edge
                    triplets.emplace_back(ii, ij, -1.0);
                } else {
                    // interior–boundary edge: move to RHS
                    rhs_u[ii] += U_out[lj];
                    rhs_v[ii] += V_out[lj];
                }
            }
        }
    }

    // Diagonal entries
    for (int k = 0; k < Ni; ++k)
        triplets.emplace_back(k, k, degree[k]);

    if (triplets.empty()) return true;

    Eigen::SparseMatrix<double> L(Ni, Ni);
    L.setFromTriplets(triplets.begin(), triplets.end());
    L.makeCompressed();

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(L);
    if (solver.info() != Eigen::Success) {
        // Fall back to identity (uniform [0,1] mapping)
        for (int k = 0; k < Ni; ++k) {
            U_out[interior[k]] = 0.5;
            V_out[interior[k]] = 0.5;
        }
        return false;
    }

    Eigen::VectorXd u_int = solver.solve(rhs_u);
    Eigen::VectorXd v_int = solver.solve(rhs_v);

    for (int k = 0; k < Ni; ++k) {
        U_out[interior[k]] = u_int[k];
        V_out[interior[k]] = v_int[k];
    }
    return true;
}

} // anonymous namespace


// ==========================================================================
// build_motorcycle_graph
// ==========================================================================

PatchLayout build_motorcycle_graph(
    const HalfEdgeMesh&                     mesh,
    const CrossField&                       field,
    const UVParam&                          uv,
    const std::vector<SingularityInfo>&     singularities,
    int                                     max_track_steps,
    int                                     /* num_threads */)
  
{

    std::vector<int> fallback_patches;
    PatchLayout layout;

    const int nv = mesh.num_vertices();
    const int nf = mesh.num_faces();
    if (nv == 0 || nf == 0) return layout;

    if (max_track_steps <= 0) max_track_steps = 10 * nv;

    // Build period jump map for seam-aware UV lookups during tracing
    PeriodJumpMap jmap = build_period_jump_map(uv);

    // ------------------------------------------------------------------
    // 1. Identify PatchVertex nodes
    // ------------------------------------------------------------------

    // Map: mesh vertex → PatchVertex index
    std::unordered_map<int,int> vert_to_patch;
    vert_to_patch.reserve(singularities.size() * 2);

    // Add singularities
    for (auto& si : singularities) {
        int pv_idx = static_cast<int>(layout.vertices.size());
        vert_to_patch[si.vertex_index] = pv_idx;

        PatchVertex pv;
        pv.mesh_vertex = si.vertex_index;
        pv.kind        = PatchVertexKind::Singularity;
        pv.position    = si.position;
        pv.uv[0]       = uv.U[si.vertex_index];
        pv.uv[1]       = uv.V[si.vertex_index];
        // −¼ singularity → valence 5 → 3 riders; +¼ → valence 3 → 1 rider
        pv.num_tracks  = (si.index < 0.0) ? 3 : 1;
        layout.vertices.push_back(pv);
    }

    // Add boundary corners (vertices with exactly one boundary edge pair
    // forming a concave or convex corner in the UV sense)
    {
        const auto& bedges = mesh.boundary_edges();
        // boundary_verts: set of boundary vertices
        std::unordered_set<int> bvert_set;
        bvert_set.reserve(bedges.size() * 2);
        for (auto& [a,b] : bedges) {
            bvert_set.insert(a);
            bvert_set.insert(b);
        }

        for (int v : bvert_set) {
            if (vert_to_patch.count(v)) continue;  // already a singularity
            // Count boundary neighbours
            int bn = 0;
            mesh.vertex_neighbours(v, [&](int nb){
                if (bvert_set.count(nb)) ++bn;
            });
            // A boundary corner has exactly 2 boundary neighbours with a
            // large angular change in UV direction.  For robustness, treat
            // any vertex with 2 boundary neighbours as a corner.
            if (bn >= 2) {
                int pv_idx = static_cast<int>(layout.vertices.size());
                vert_to_patch[v] = pv_idx;

                PatchVertex pv;
                pv.mesh_vertex = v;
                pv.kind        = PatchVertexKind::BoundaryCorner;
                pv.position    = mesh.vertex(v).pos;
                pv.uv[0]       = uv.U[v];
                pv.uv[1]       = uv.V[v];
                pv.num_tracks  = 1;
                layout.vertices.push_back(pv);
            }
        }
    }

    if (layout.vertices.empty()) {
        // No singularities or boundary corners — the mesh is a closed genus-0
        // surface with a perfect 4-RoSy field.  No motorcycle tracks needed;
        // the entire mesh forms one patch.
        PatchFace sole;
        sole.n_corners = 0;
        sole.contained_faces.resize(nf);
        std::iota(sole.contained_faces.begin(), sole.contained_faces.end(), 0);
        sole.target_quads_u = 4;
        sole.target_quads_v = 4;
        layout.faces.push_back(std::move(sole));
        return layout;
    }

    // ------------------------------------------------------------------
    // 2. Seed and advance riders
    // ------------------------------------------------------------------

    // track_mark[fi] = index of the first PatchEdge that crosses face fi
    // (used to halt riders when they enter an already-tracked region)
    std::vector<int> face_track_mark(nf, -1);

    // edge_track_mark: directed edge → track index
    std::unordered_map<int64_t, int> edge_track_mark;
    edge_track_mark.reserve(nf * 3);

    std::vector<Rider> riders;

    for (int pvi = 0; pvi < static_cast<int>(layout.vertices.size()); ++pvi) {
        const PatchVertex& pv = layout.vertices[pvi];
        int v = pv.mesh_vertex;

        // Seed riders from all faces adjacent to this PatchVertex
        // For each adjacent face, spawn a rider in the dominant UV direction.
        std::unordered_set<int> seeded_faces;
        mesh.vertex_faces(v, [&](int fi) {
            if (seeded_faces.count(fi)) return;
            seeded_faces.insert(fi);

            auto [du, dv] = face_uv_gradient(fi, mesh, uv);
            RiderDir rdir = (du >= dv) ? RiderDir::V_CONST : RiderDir::U_CONST;
            double iso_val = (rdir == RiderDir::U_CONST)
                ? std::round(uv.U[v]) : std::round(uv.V[v]);

            Rider r;
            r.patch_vertex_idx = pvi;
            r.current_face     = fi;
            r.src_v            = v;
            r.iso_value        = iso_val;
            r.dir              = rdir;
            r.track.push_back(v);
            riders.push_back(std::move(r));
        });
    }

    // Advance each rider
    for (auto& rider : riders) {
        if (rider.halted) continue;

        int entry_va = -1, entry_vb = -1;
        int steps = 0;

        while (!rider.halted && steps < max_track_steps) {
            ++steps;

            int fi = rider.current_face;
            if (fi < 0) { rider.halted = true; break; }

            // Halt if this face is already claimed by a different track
            if (face_track_mark[fi] >= 0) {
                int other_track = face_track_mark[fi];
                (void)other_track;
                rider.halted = true;
                break;
            }

            EdgeCrossing ec;
            bool found = find_exit_edge(
                fi, rider.iso_value, rider.dir,
                entry_va, entry_vb,
                mesh, uv, jmap, ec);

            if (!found) {
                rider.halted = true;
                break;
            }

            // Mark this face as traversed by this rider
            face_track_mark[fi] = static_cast<int>(layout.edges.size());

            // Record the track vertex
            if (ec.track_vert != rider.src_v)
                rider.track.push_back(ec.track_vert);

            // Check: is the destination a known PatchVertex?
            auto dest_it = vert_to_patch.find(ec.track_vert);
            if (dest_it != vert_to_patch.end() &&
                dest_it->second != rider.patch_vertex_idx) {
                // Rider reached a PatchVertex — create the PatchEdge and stop
                rider.halted = true;

                PatchEdge pe;
                pe.source          = rider.patch_vertex_idx;
                pe.target          = dest_it->second;
                pe.twin            = -1;
                pe.track_verts     = rider.track;
                pe.on_boundary     = mesh.is_boundary_vertex(ec.track_vert);
                pe.uv_steps        = static_cast<int>(
                    std::round(std::abs(rider.iso_value)));
                layout.edges.push_back(std::move(pe));
                break;
            }

            // Halt at mesh boundary
            if (ec.twin_face < 0) {
                rider.halted = true;
                if (!rider.track.empty()) {
                    PatchEdge pe;
                    pe.source      = rider.patch_vertex_idx;
                    pe.target      = -1;  // boundary — no target PatchVertex
                    pe.twin        = -1;
                    pe.track_verts = rider.track;
                    pe.on_boundary = true;
                    layout.edges.push_back(std::move(pe));
                }
                break;
            }

            // Continue into adjacent face
            entry_va = ec.va;
            entry_vb = ec.vb;
            rider.current_face = ec.twin_face;
            rider.src_v = ec.track_vert;
        }

        if (!rider.halted && !rider.track.empty()) {
            // Step-limit reached — emit a partial edge
            PatchEdge pe;
            pe.source      = rider.patch_vertex_idx;
            pe.target      = -1;
            pe.twin        = -1;
            pe.track_verts = rider.track;
            pe.on_boundary = false;
            layout.edges.push_back(std::move(pe));
        }
    }

    // ------------------------------------------------------------------
    // 3. Flood-fill patch faces from the track network
    // ------------------------------------------------------------------
    {
        std::vector<int> face_patch(nf, -1);  // face → patch index

        for (int start_face = 0; start_face < nf; ++start_face) {
            if (face_patch[start_face] >= 0) continue;
            if (face_track_mark[start_face] >= 0) continue;

            int pi = static_cast<int>(layout.faces.size());
            PatchFace pf;
            pf.n_corners = 0;

            // BFS flood fill within patch (not crossing track edges)
            std::queue<int> q;
            q.push(start_face);
            face_patch[start_face] = pi;
            pf.contained_faces.push_back(start_face);

            while (!q.empty()) {
    int fi = q.front(); q.pop();

    auto [he0, he1, he2] = mesh.face_half_edges(fi);
    const int hes[3] = {he0, he1, he2};

    for (int hei : hes) {
        int twin_hei = mesh.half_edge(hei).twin;
        if (twin_hei < 0) continue;

        int fj = mesh.half_edge(twin_hei).face;
        if (fj < 0) continue;
        if (face_patch[fj] >= 0) continue;
        if (face_track_mark[fj] >= 0) continue;

        face_patch[fj] = pi;
        pf.contained_faces.push_back(fj);
        q.push(fj);
    }
}

// ----------------------------------------------------------
// Validate patch boundary closure
// ----------------------------------------------------------

bool patch_closed = true;

{
    std::unordered_map<int,int> boundary_degree;

   for (int fi : pf.contained_faces)
{
    auto [he0, he1, he2] = mesh.face_half_edges(fi);
    const int hes[3] = {he0, he1, he2};

    for (int hei : hes)
    {
        int twin_hei = mesh.half_edge(hei).twin;

        bool is_boundary = false;

        if (twin_hei < 0)
        {
            is_boundary = true;
        }
        else
        {
            int fj = mesh.half_edge(twin_hei).face;

            // motorcycle track boundary
            if (fj >= 0 && face_track_mark[fj] >= 0)
            {
                is_boundary = true;
            }
            // neighbour belongs to another patch
            else if (fj >= 0 &&
                     face_patch[fj] >= 0 &&
                     face_patch[fj] != pi)
            {
                is_boundary = true;
            }
        }

        if (!is_boundary)
            continue;

        int prev_he = mesh.half_edge(hei).prev;
        if (prev_he < 0)
            continue;

        int va = mesh.half_edge(prev_he).vertex;
        int vb = mesh.half_edge(hei).vertex;

        boundary_degree[va]++;
        boundary_degree[vb]++;
    }
}

    for (auto& kv : boundary_degree)
    {
        if (kv.second != 2)
        {
            patch_closed = false;
            break;
        }
    }
}

if (!patch_closed)
{
    fallback_patches.push_back(pi);
}
            // Find corner PatchVertices on the boundary of this patch
            std::unordered_set<int> patch_face_set(
                pf.contained_faces.begin(), pf.contained_faces.end());

            std::unordered_set<int> corner_set;
            for (int fi : pf.contained_faces) {
                auto [v0, v1, v2] = mesh.face_vertices(fi);
                for (int v : {v0, v1, v2}) {
                    auto it = vert_to_patch.find(v);
                    if (it != vert_to_patch.end()) corner_set.insert(v);
                }
            }

            std::vector<int> corner_verts(corner_set.begin(), corner_set.end());
            pf.n_corners = static_cast<int>(
                std::min(corner_verts.size(), size_t(PatchFace::MAX_CORNERS)));

            for (int k = 0; k < pf.n_corners; ++k)
                pf.corners[k] = vert_to_patch[corner_verts[k]];

            // Compute target quad resolution from patch area
            double area = 0.0;
            for (int fi : pf.contained_faces)
                area += mesh.face_area(fi);

            // Use sqrt(area) as a proxy for linear dimension → quad count
            pf.target_quads_u = std::max(1, static_cast<int>(std::ceil(std::sqrt(area) * 4)));
            pf.target_quads_v = pf.target_quads_u;

            layout.faces.push_back(std::move(pf));
        }
    }

    // Update statistics
    layout.n_degenerate = 0;
    double total_aspect = 0.0;
    for (auto& pf : layout.faces) {
        if (pf.is_degenerate()) ++layout.n_degenerate;
        // Simple aspect estimate: face count / expected count for a square patch
        int n = static_cast<int>(pf.contained_faces.size());
        double aspect = (n > 0) ? std::sqrt(static_cast<double>(n)) : 1.0;
        total_aspect += aspect;
        layout.n_quads_total += pf.target_quads_u * pf.target_quads_v;
    }
    if (!layout.faces.empty())
        layout.mean_aspect = total_aspect / layout.faces.size();

    return layout;
}


// ==========================================================================
// build_patch_layout  (public entry point — wraps build_motorcycle_graph)
// ==========================================================================

PatchLayout build_patch_layout(
    const HalfEdgeMesh&                     mesh,
    const CrossField&                       field,
    const UVParam&                          uv,
    const std::vector<SingularityInfo>&     singularities,
    int                                     target_quad_count,
    int                                     num_threads)
{
    PatchLayout layout = build_motorcycle_graph(
        mesh, field, uv, singularities, -1, num_threads);

    // Scale per-patch target quad counts to hit the global target
    if (layout.n_quads_total > 0 && target_quad_count > 0) {
        double scale = static_cast<double>(target_quad_count)
                     / static_cast<double>(layout.n_quads_total);
        for (auto& pf : layout.faces) {
            int bu = std::max(1, static_cast<int>(
                std::round(pf.target_quads_u * std::sqrt(scale))));
            int bv = std::max(1, static_cast<int>(
                std::round(pf.target_quads_v * std::sqrt(scale))));
            pf.target_quads_u = bu;
            pf.target_quads_v = bv;
        }
    }

    return layout;
}


// ==========================================================================
// extract_quads_motorcycle — full method=1 pipeline
// ==========================================================================

QuadMesh extract_quads_motorcycle(
    const HalfEdgeMesh&                     mesh,
    const CrossField&                       field,
    const UVParam&                          uv,
    const std::vector<SingularityInfo>&     singularities,
    int                                     target_quad_count,
    int                                     num_threads)
{
    const int nv = mesh.num_vertices();
    const int nf = mesh.num_faces();

    // Build patch layout
    PatchLayout layout = build_patch_layout(
        mesh, field, uv, singularities, target_quad_count, num_threads);

    if (layout.faces.empty()) {
        // Fallback to global isoline extraction
        return extract_quads_from_isolines_seam(mesh, uv, num_threads);
    }

    // Prepare global vertex position lookup
    const auto& mesh_verts = mesh.vertices();

    // Output accumulator
    QuadMesh global;
    global.vertices.reserve(target_quad_count * 2);
    global.quads.reserve(target_quad_count);

    // Vertex offset for merging per-patch results
    int vert_offset = 0;

    for (const PatchFace& pf : layout.faces) {
        if (pf.contained_faces.empty()) continue;

        // ----------------------------------------------------------------
        // Step A: Gather unique vertices for this patch
        // ----------------------------------------------------------------
        std::unordered_map<int,int> global_to_local;
        global_to_local.reserve(pf.contained_faces.size() * 3);

        std::vector<int> local_to_global;  // local idx → mesh vertex idx
        local_to_global.reserve(pf.contained_faces.size() * 3);

        std::vector<std::array<int,3>> local_faces;
        local_faces.reserve(pf.contained_faces.size());

        for (int fi : pf.contained_faces) {
            auto [v0, v1, v2] = mesh.face_vertices(fi);
            std::array<int,3> lf;
            for (int k = 0; k < 3; ++k) {
                int gv = (k==0 ? v0 : k==1 ? v1 : v2);
                auto [it, inserted] = global_to_local.emplace(
                    gv, static_cast<int>(local_to_global.size()));
                if (inserted) local_to_global.push_back(gv);
                lf[k] = it->second;
            }
            local_faces.push_back(lf);
        }

        const int lnv = static_cast<int>(local_to_global.size());
        if (lnv < 3) continue;

        // ----------------------------------------------------------------
        // Step B: For degenerate patches, use global UV directly
        // ----------------------------------------------------------------
        if (pf.is_degenerate() || pf.n_corners < 4) {
            // Build local UVParam from global UV
            UVParam local_uv;
            local_uv.U.resize(lnv);
            local_uv.V.resize(lnv);
            for (int li = 0; li < lnv; ++li) {
                int gi = local_to_global[li];
                local_uv.U[li] = uv.U[gi];
                local_uv.V[li] = uv.V[gi];
            }

            // Build local HalfEdgeMesh
            std::vector<Vec3> lpos(lnv);
            for (int li = 0; li < lnv; ++li)
                lpos[li] = mesh_verts[local_to_global[li]].pos;

            HalfEdgeMesh local_mesh(lpos, local_faces);
            QuadMesh patch_qm = extract_quads_from_isolines(
                local_mesh, local_uv, num_threads);

            // Remap local vertices to global world positions
            for (auto& v : patch_qm.vertices) {
                // v is already in local (patch) 3-D space — just reindex
                (void)v;
            }

            // Merge into global result with offset
            int base = static_cast<int>(global.vertices.size());
            for (auto& v : patch_qm.vertices) global.vertices.push_back(v);
            for (auto& q : patch_qm.quads) {
                global.quads.push_back({q[0]+base, q[1]+base,
                                        q[2]+base, q[3]+base});
            }
            for (auto& t : patch_qm.tris) {
                global.tris.push_back({t[0]+base, t[1]+base, t[2]+base});
            }
            vert_offset = static_cast<int>(global.vertices.size());
            continue;
        }

        // ----------------------------------------------------------------
        // Step C: Identify boundary vertices and map to [0,1]^2
        // ----------------------------------------------------------------
        // Find the 4 corner local indices (PatchVertex corners for this patch)
        std::vector<int> boundary_local_corners;
        for (int k = 0; k < pf.n_corners && k < 4; ++k) {
            int pvi = pf.corners[k];
            if (pvi < 0 || pvi >= (int)layout.vertices.size()) continue;
            int mesh_v = layout.vertices[pvi].mesh_vertex;
            auto it = global_to_local.find(mesh_v);
            if (it != global_to_local.end())
                boundary_local_corners.push_back(it->second);
        }

        // If we can't find all 4 corners in this patch, fall back to global UV
        if (static_cast<int>(boundary_local_corners.size()) < 4) {
            UVParam local_uv;
            local_uv.U.resize(lnv);
            local_uv.V.resize(lnv);
            for (int li = 0; li < lnv; ++li) {
                local_uv.U[li] = uv.U[local_to_global[li]];
                local_uv.V[li] = uv.V[local_to_global[li]];
            }
            std::vector<Vec3> lpos(lnv);
            for (int li = 0; li < lnv; ++li)
                lpos[li] = mesh_verts[local_to_global[li]].pos;

            HalfEdgeMesh local_mesh(lpos, local_faces);
            QuadMesh patch_qm = extract_quads_from_isolines(
                local_mesh, local_uv, num_threads);

            int base = static_cast<int>(global.vertices.size());
            for (auto& v : patch_qm.vertices) global.vertices.push_back(v);
            for (auto& q : patch_qm.quads)
                global.quads.push_back({q[0]+base,q[1]+base,q[2]+base,q[3]+base});
            for (auto& t : patch_qm.tris)
                global.tris.push_back({t[0]+base,t[1]+base,t[2]+base});
            vert_offset = static_cast<int>(global.vertices.size());
            continue;
        }

        // Map corners to (0,0), (1,0), (1,1), (0,1)
        const std::pair<double,double> corner_uvs[4] = {
            {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}
        };

        // Collect all boundary local vertex indices (corner vertices only
        // for simplicity — full boundary ordering requires more topology work)
        std::vector<int> boundary_local;
        std::vector<std::pair<double,double>> boundary_uv_vals;
        for (int k = 0; k < 4; ++k) {
            boundary_local.push_back(boundary_local_corners[k]);
            boundary_uv_vals.push_back(corner_uvs[k]);
        }

        // ----------------------------------------------------------------
        // Step D: Solve Tutte parametrization
        // ----------------------------------------------------------------
        std::vector<double> U_patch, V_patch;
        bool solved = solve_tutte(
            local_to_global,   // sub_verts (used for size)
            local_faces,
            boundary_local,
            boundary_uv_vals,
            U_patch, V_patch);

        if (!solved) {
            // Fallback: use raw UV
            U_patch.resize(lnv); V_patch.resize(lnv);
            for (int li = 0; li < lnv; ++li) {
                U_patch[li] = uv.U[local_to_global[li]];
                V_patch[li] = uv.V[local_to_global[li]];
            }
        }

        // Scale to target resolution
        for (int li = 0; li < lnv; ++li) {
            U_patch[li] *= pf.target_quads_u;
            V_patch[li] *= pf.target_quads_v;
        }

        // ----------------------------------------------------------------
        // Step E: Isoline extraction on this patch
        // ----------------------------------------------------------------
        UVParam patch_uv;
        patch_uv.U = Eigen::Map<Eigen::VectorXd>(U_patch.data(), lnv);
        patch_uv.V = Eigen::Map<Eigen::VectorXd>(V_patch.data(), lnv);

        std::vector<Vec3> lpos(lnv);
        for (int li = 0; li < lnv; ++li)
            lpos[li] = mesh_verts[local_to_global[li]].pos;

        HalfEdgeMesh local_mesh(lpos, local_faces);
        QuadMesh patch_qm = extract_quads_from_isolines(
            local_mesh, patch_uv, num_threads);

        // ----------------------------------------------------------------
        // Step F: Merge patch result into global mesh
        // ----------------------------------------------------------------
        int base = static_cast<int>(global.vertices.size());
        for (auto& v : patch_qm.vertices) global.vertices.push_back(v);
        for (auto& q : patch_qm.quads)
            global.quads.push_back({q[0]+base, q[1]+base,
                                    q[2]+base, q[3]+base});
        for (auto& t : patch_qm.tris)
            global.tris.push_back({t[0]+base, t[1]+base, t[2]+base});

        vert_offset = static_cast<int>(global.vertices.size());
    }

    // ----------------------------------------------------------------
    // Step G: Global vertex weld (merges shared boundary track vertices)
    // ----------------------------------------------------------------
    if (!global.vertices.empty()) {
        // BUG FIX (v92): weld_vertices() and compact_vertices() were updated
        // in v91 to take std::vector<double>& (previously std::vector<float>&).
        // This call site still passed std::vector<float>, causing a compile
        // error with the v91 API.  Build the position buffer as double from
        // the start to match the current function signatures.
        std::vector<double> pos_d;
        pos_d.reserve(global.vertices.size() * 3);
        for (auto& v : global.vertices)
            for (int k = 0; k < 3; ++k) pos_d.push_back(v[k]);

        std::vector<int32_t> qf_out, tf_out;
        qf_out.reserve(global.quads.size() * 4);
        tf_out.reserve(global.tris.size() * 3);
        for (auto& q : global.quads)
            for (int k : q) qf_out.push_back(k);
        for (auto& t : global.tris)
            for (int k : t) tf_out.push_back(k);

        int32_t nv_out = static_cast<int32_t>(global.vertices.size());
        weld_vertices(pos_d, qf_out, tf_out, nv_out, 1e-5);
        compact_vertices(pos_d, qf_out, tf_out, nv_out);

        // Rebuild global from welded arrays
        global.vertices.resize(nv_out);
        for (int i = 0; i < nv_out; ++i)
            global.vertices[i] = {pos_d[3*i], pos_d[3*i+1], pos_d[3*i+2]};

        int nq = static_cast<int>(qf_out.size()) / 4;
        global.quads.resize(nq);
        for (int i = 0; i < nq; ++i)
            global.quads[i] = {qf_out[4*i], qf_out[4*i+1],
                                qf_out[4*i+2], qf_out[4*i+3]};

        int nt = static_cast<int>(tf_out.size()) / 3;
        global.tris.resize(nt);
        for (int i = 0; i < nt; ++i)
            global.tris[i] = {tf_out[3*i], tf_out[3*i+1], tf_out[3*i+2]};
    }

    // Update statistics
    int total = static_cast<int>(global.quads.size() + global.tris.size());
    global.quad_percentage = total > 0
        ? 100.f * static_cast<float>(global.quads.size()) / total : 0.f;

    return global;
}


// ==========================================================================
// patch_aspect_ratio
// ==========================================================================

double patch_aspect_ratio(const PatchLayout& layout, int fi)
{
    if (fi < 0 || fi >= static_cast<int>(layout.faces.size()))
        return 1.0;

    const PatchFace& pf = layout.faces[fi];
    if (pf.n_corners < 4) return 1.0;

    // Compute arc-lengths of the 4 bounding edges
    std::vector<double> lens;
    for (int k = 0; k < pf.n_corners && k < 4; ++k) {
        int ei = pf.edges[k];
        if (ei < 0 || ei >= static_cast<int>(layout.edges.size())) continue;
        lens.push_back(layout.edges[ei].arc_length);
    }
    if (lens.size() < 4) return 1.0;

    double side_u = 0.5 * (lens[0] + lens[2]);  // average of opposite sides
    double side_v = 0.5 * (lens[1] + lens[3]);

    if (side_v < 1e-12) return 1.0;
    double ratio = side_u / side_v;
    return ratio < 1.0 ? 1.0 / ratio : ratio;
}


// ==========================================================================
// patch_quad_budget
// ==========================================================================

int patch_quad_budget(
    const PatchLayout& layout,
    int                fi,
    double             total_mesh_area,
    int                global_target,
    int                max_quads_per_patch)
{
    if (fi < 0 || fi >= static_cast<int>(layout.faces.size()))
        return 1;
    if (total_mesh_area <= 0.0 || global_target <= 0)
        return 1;

    // Estimate patch area from its precomputed face list
    // We can't call mesh.face_area() here since we don't have the mesh,
    // so use the target_quads field as area proxy.
    const PatchFace& pf = layout.faces[fi];
    int base = pf.target_quads_u * pf.target_quads_v;
    int budget = std::max(1, std::min(max_quads_per_patch,
        static_cast<int>(std::round(
            static_cast<double>(global_target) *
            static_cast<double>(base) /
            static_cast<double>(std::max(1, layout.n_quads_total))))));
    return budget;
}

} // namespace qf
