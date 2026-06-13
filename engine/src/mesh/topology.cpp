/**
 * topology.cpp — Topology analysis for QuadForge.
 */

#include "../../include/quadforge/mesh/topology.h"
#include <sstream>
#include <queue>
#include <set>
#include <unordered_map>
#include <array>

namespace qf {

TopologyReport analyse_topology(const HalfEdgeMesh& mesh) {
    TopologyReport r;
    r.num_vertices   = mesh.num_vertices();
    r.num_faces      = mesh.num_faces();
    r.num_edges      = 0; // counted below

    // Count unique undirected edges.
    //
    // We use the identity  E = V − χ + F  derived from χ = V − E + F.
    // This is always correct by definition of the Euler characteristic,
    // regardless of whether the mesh is manifold, open, or has multiple
    // connected components — the algebraic identity holds for any CW complex.
    //
    // BUG FIX (v103): The previous comment here suggested deriving E as
    // "(nhe - boundary_count) / 2 + boundary_count" which over-counts edges
    // incident on 3+ half-edges (non-manifold).  The Euler-formula derivation
    // is both simpler and always correct.
    //
    // Note: mesh.euler_characteristic() calls edge_map_.size() implicitly via
    //   χ = V − E + F  →  E = V − χ + F
    // so this is consistent with the true undirected edge count stored in
    // the half-edge structure's edge_map_.
    r.euler_characteristic = mesh.euler_characteristic();
    r.num_edges = r.num_vertices - r.euler_characteristic + r.num_faces;

    r.genus                = mesh.genus();
    r.num_boundary_loops   = mesh.num_boundary_loops();
    r.is_closed            = mesh.is_closed();
    r.is_manifold          = mesh.is_manifold();
    r.non_manifold_edges   = mesh.non_manifold_edges();
    r.num_non_manifold_edges = (int)r.non_manifold_edges.size();

    // Non-manifold vertices: a vertex is non-manifold when its incident face
    // fan consists of more than one connected component (the ring walk from
    // the stored half_edge only visits ONE component).
    //
    // Detection strategy:
    //   1. Pre-compute per-vertex total face count via one O(F) pass.
    //   2. For each vertex, count faces reachable by vertex_ring (one component).
    //   3. If ring_faces < total_faces the vertex has disconnected components
    //      => non-manifold.
    //
    // Bug fixed (v68): the previous code checked `ring_faces == 0`, which is
    // never true for a non-isolated vertex — the ring walk always finds at
    // least one face for any vertex whose stored half_edge is valid.  That
    // made the non-manifold vertex counter permanently stuck at 0.

    // Step 1: O(F) pass — tally how many faces reference each vertex.
    std::vector<int> vtx_face_count(mesh.num_vertices(), 0);
    for (int fi = 0; fi < mesh.num_faces(); ++fi) {
        auto verts = mesh.face_vertices(fi);
        for (int v : verts)
            if (v >= 0 && v < mesh.num_vertices())
                ++vtx_face_count[v];
    }

    // Step 2: per-vertex ring walk + comparison.
    r.num_non_manifold_vertices = 0;
    for (int vi = 0; vi < mesh.num_vertices(); ++vi) {
        int start_he = mesh.vertex(vi).half_edge;
        if (start_he < 0) continue; // isolated vertex — skip

        // Count faces reachable through the continuous ring walk (one fan).
        int ring_faces = 0;
        mesh.vertex_ring(vi, [&](int he){
            if (mesh.half_edge(he).face >= 0) ++ring_faces;
        });

        // If ring_faces < total count, at least one face component is
        // disconnected from the primary fan => non-manifold vertex.
        if (ring_faces < vtx_face_count[vi]) {
            ++r.num_non_manifold_vertices;
        }
    }

    // Connected components via BFS over face adjacency graph.
    // BUG FIX (v105): The original BFS only traversed face-connected components
    // and never counted isolated vertices (vertices whose stored half_edge == -1,
    // i.e. vertices not referenced by any face).  This made the reported
    // r.num_connected_components inconsistent with the value returned by
    // HalfEdgeMesh::num_connected_components() (which calls
    // num_connected_components_internal() and DOES count isolated vertices).
    // Any caller that compared or subtracted the two values would get a
    // different result depending on which function it called, silently
    // corrupting topology-based computations that depend on the component count.
    //
    // Fix: after the face-adjacency BFS, perform a second O(V) pass that
    // increments `comps` once for each vertex with no incident face — matching
    // the identical logic at the bottom of num_connected_components_internal().
    std::vector<bool> visited(mesh.num_faces(), false);
    int comps = 0;
    for (int fi = 0; fi < mesh.num_faces(); ++fi) {
        if (visited[fi]) continue;
        ++comps;
        std::queue<int> q;
        q.push(fi);
        while (!q.empty()) {
            int f = q.front(); q.pop();
            if (visited[f]) continue;
            visited[f] = true;
            // Flood fill via adjacent faces through half-edge twins
            auto hes = mesh.face_half_edges(f);
            for (int he : hes) {
                int tw = mesh.half_edge(he).twin;
                if (tw >= 0) {
                    int nf = mesh.half_edge(tw).face;
                    if (nf >= 0 && !visited[nf]) q.push(nf);
                }
            }
        }
    }

    // Count isolated vertices (not referenced by any face).
    // These are their own components and must be added to the face-BFS count.
    for (int vi = 0; vi < mesh.num_vertices(); ++vi)
        if (mesh.vertex(vi).half_edge < 0) ++comps;

    r.num_connected_components = comps;

    return r;
}

std::string TopologyReport::to_string() const {
    std::ostringstream os;
    os << "[Topology] V=" << num_vertices
       << " E=" << num_edges
       << " F=" << num_faces
       << " χ=" << euler_characteristic
       << " genus=" << genus
       << " boundary_loops=" << num_boundary_loops
       << " components=" << num_connected_components
       << (is_manifold ? " manifold" : " NON-MANIFOLD")
       << (is_closed   ? " closed"   : " open");
    if (num_non_manifold_edges > 0)
        os << " non_manifold_edges=" << num_non_manifold_edges;
    if (num_non_manifold_vertices > 0)
        os << " non_manifold_verts=" << num_non_manifold_vertices;
    return os.str();
}

int repair_non_manifold(HalfEdgeMesh& mesh) {
    // Strategy: collect the set of non-manifold edges; remove every face that
    // contains one of those edges; rebuild the HalfEdgeMesh from the surviving
    // faces and compact the vertex list.  Returns the number of faces removed.
    //
    // This is a conservative approach (may over-remove), but it guarantees
    // the resulting mesh is manifold and avoids undefined behaviour in the
    // downstream pipeline stages.

    if (mesh.is_manifold()) return 0;

    const auto& nm_edges = mesh.non_manifold_edges();
    if (nm_edges.empty()) return 0;

    // Build a set of vertex-pairs for fast lookup
    std::set<std::pair<int,int>> bad_edge_set(nm_edges.begin(), nm_edges.end());

    int nf = mesh.num_faces();
    int nv = mesh.num_vertices();

    // Find all faces that contain a non-manifold edge
    std::vector<bool> face_bad(nf, false);
    for (int fi = 0; fi < nf; ++fi) {
        auto verts = mesh.face_vertices(fi);
        for (int k = 0; k < 3; ++k) {
            int a = verts[k], b = verts[(k+1)%3];
            if (bad_edge_set.count({std::min(a,b), std::max(a,b)})) {
                face_bad[fi] = true;
                break;
            }
        }
    }

    // Count how many faces will be removed
    int removed = 0;
    for (int fi = 0; fi < nf; ++fi) removed += face_bad[fi] ? 1 : 0;
    if (removed == 0) return 0;

    // Build new face list (using original vertex indices)
    std::vector<std::array<int,3>> new_faces;
    new_faces.reserve(nf - removed);
    for (int fi = 0; fi < nf; ++fi) {
        if (!face_bad[fi])
            new_faces.push_back(mesh.face_vertices(fi));
    }

    // Compact vertex list: only keep vertices referenced by surviving faces
    std::vector<int> old_to_new(nv, -1);
    std::vector<Vec3> new_positions;
    new_positions.reserve(nv);
    for (auto& f : new_faces) {
        for (int k = 0; k < 3; ++k) {
            int v = f[k];
            if (old_to_new[v] < 0) {
                old_to_new[v] = (int)new_positions.size();
                new_positions.push_back(mesh.vertex_pos(v));
            }
            f[k] = old_to_new[v];
        }
    }

    // Rebuild mesh in-place via move assignment (all members are std::vector,
    // so the compiler-generated move-assign is safe and efficient)
    mesh = HalfEdgeMesh(new_positions, new_faces);
    return removed;
}

std::vector<int> connected_components(const HalfEdgeMesh& mesh) {
    int nv = mesh.num_vertices();
    std::vector<int> comp(nv, -1);
    int id = 0;
    for (int vi = 0; vi < nv; ++vi) {
        if (comp[vi] >= 0) continue;
        std::queue<int> q;
        q.push(vi);
        while (!q.empty()) {
            int v = q.front(); q.pop();
            if (comp[v] >= 0) continue;
            comp[v] = id;
            mesh.vertex_neighbours(v, [&](int nb){
                if (comp[nb] < 0) q.push(nb);
            });
        }
        ++id;
    }
    return comp;
}

} // namespace qf
