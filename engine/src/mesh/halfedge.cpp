/**
 * halfedge.cpp — Half-edge mesh implementation.
 *
 * Builds a manifold half-edge mesh from flat vertex/face arrays in O(V+F).
 * Supports traversal, topology queries, and geometry helpers used by
 * every subsequent pipeline stage.
 */

#include "../../include/quadforge/mesh/halfedge.h"

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <sstream>
#include <array>
#include <functional>

namespace qf {

// -----------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------

HalfEdgeMesh::HalfEdgeMesh(const std::vector<Vec3>& positions,
                             const std::vector<std::array<int,3>>& tri_faces)
{
    // Allocate vertices
    vertices_.resize(positions.size());
    for (int i = 0; i < (int)positions.size(); ++i)
        vertices_[i].pos = positions[i];

    // Allocate faces
    faces_.resize(tri_faces.size());

    // Allocate half-edges: 3 per face
    int nf = (int)tri_faces.size();
    half_edges_.resize(3 * nf);

    // First pass: fill vertex / next / prev for each face's 3 half-edges
    for (int fi = 0; fi < nf; ++fi) {
        faces_[fi].half_edge = fi * 3;
        for (int k = 0; k < 3; ++k) {
            int he = fi * 3 + k;
            half_edges_[he].vertex = tri_faces[fi][(k + 1) % 3]; // destination vertex
            half_edges_[he].face   = fi;
            half_edges_[he].next   = fi * 3 + (k + 1) % 3;
            half_edges_[he].prev   = fi * 3 + (k + 2) % 3;
            half_edges_[he].twin   = -1; // filled in next pass
        }
    }

    // Second pass: match twins using edge map
    // Key: (v_lo, v_hi) → list of half-edge indices with that undirected edge
    edge_map_.clear();
    for (int fi = 0; fi < nf; ++fi) {
        for (int k = 0; k < 3; ++k) {
            int he = fi * 3 + k;
            int from = tri_faces[fi][k];
            int to   = tri_faces[fi][(k + 1) % 3];
            int lo = std::min(from, to), hi = std::max(from, to);
            edge_map_[{lo, hi}].push_back(he);
        }
    }

    for (auto& [edge, hes] : edge_map_) {
        if (hes.size() == 2) {
            half_edges_[hes[0]].twin = hes[1];
            half_edges_[hes[1]].twin = hes[0];
        }
        // > 2 half-edges on same undirected edge → non-manifold
    }

    // Assign one outgoing half-edge per vertex
    for (int fi = 0; fi < nf; ++fi) {
        for (int k = 0; k < 3; ++k) {
            int he   = fi * 3 + k;
            int from = tri_faces[fi][k];
            if (vertices_[from].half_edge == -1)
                vertices_[from].half_edge = he;
        }
    }

    build_connectivity();
    detect_topology_issues();
}

HalfEdgeMesh HalfEdgeMesh::from_matrices(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{
    std::vector<Vec3> positions(V.rows());
    for (int i = 0; i < V.rows(); ++i)
        positions[i] = V.row(i).transpose();

    std::vector<std::array<int,3>> faces(F.rows());
    for (int i = 0; i < F.rows(); ++i)
        faces[i] = {F(i,0), F(i,1), F(i,2)};

    return HalfEdgeMesh(positions, faces);
}

// -----------------------------------------------------------------------
// build_connectivity — detect boundary loops
// -----------------------------------------------------------------------

void HalfEdgeMesh::build_connectivity() {
    boundary_edges_.clear();
    for (auto& [edge, hes] : edge_map_) {
        if (hes.size() == 1)
            boundary_edges_.push_back(edge);
    }
}

void HalfEdgeMesh::detect_topology_issues() {
    non_manifold_edges_.clear();
    for (auto& [edge, hes] : edge_map_) {
        if (hes.size() > 2)
            non_manifold_edges_.push_back(edge);
    }
}

// -----------------------------------------------------------------------
// Traversal
// -----------------------------------------------------------------------

void HalfEdgeMesh::vertex_ring(int v, std::function<void(int)> cb) const {
    int start = vertices_[v].half_edge;
    if (start < 0) return;

    // Forward traversal (CCW) until we complete the loop or hit a boundary.
    int he = start;
    int guard = 0;
    int max_iter = static_cast<int>(half_edges_.size()) + 4;
    bool hit_boundary = false;

    do {
        cb(he);
        int tw = half_edges_[he].twin;
        if (tw < 0) { hit_boundary = true; break; }
        he = half_edges_[tw].next;
        if (++guard > max_iter) break;
    } while (he != start);

    // If we hit the boundary going CCW, also traverse CW from start so that
    // boundary vertices with a fan of faces are fully covered.
    // Backward step: for the outgoing he 'rev', the CW next-outgoing is
    //   twin( prev(rev) )
    if (hit_boundary) {
        int rev = start;
        guard = 0;
        while (true) {
            int prev_he = half_edges_[rev].prev;
            int prev_tw = half_edges_[prev_he].twin;
            if (prev_tw < 0) break; // boundary on the other side — done
            rev = prev_tw;
            if (rev == start) break; // full loop (shouldn't happen for open boundary)
            if (++guard > max_iter) break;
            cb(rev);
        }
    }
}

void HalfEdgeMesh::vertex_neighbours(int v, std::function<void(int)> cb) const {
    vertex_ring(v, [&](int he){
        cb(half_edges_[he].vertex);
    });
}

void HalfEdgeMesh::vertex_faces(int v, std::function<void(int)> cb) const {
    vertex_ring(v, [&](int he){
        if (half_edges_[he].face >= 0) cb(half_edges_[he].face);
    });
}

// edge_ring — alias of vertex_ring with roadmap-matching name.
void HalfEdgeMesh::edge_ring(int v, std::function<void(int he)> cb) const {
    vertex_ring(v, cb);
}

// -----------------------------------------------------------------------
// boundary_loop — walk one boundary loop given a starting boundary he.
// -----------------------------------------------------------------------
void HalfEdgeMesh::boundary_loop(int start_he,
                                  std::function<void(int he)> cb) const
{
    if (start_he < 0 || start_he >= (int)half_edges_.size()) return;
    // A boundary half-edge has twin == -1.
    //
    // To advance to the NEXT boundary half-edge going CCW around the hole
    // we walk FORWARD along the face (using .next), which takes us from
    // the tail vertex of `he` to the next boundary edge leaving the head
    // vertex of `he`:
    //
    //   boundary_next_ccw(he):
    //     cur = he.next           -- advance within the owning face
    //     while cur.twin != -1:   -- if interior, jump to the adjacent face
    //         cur = cur.twin.next
    //     return cur              -- found the next boundary segment
    //
    // NOTE: the old implementation used .prev chains, which traversed the
    // boundary in the CW direction (reversed order).  The .next chain
    // produces the correct CCW traversal consistent with standard
    // half-edge conventions for outward-oriented meshes.

    auto boundary_next = [&](int he) -> int {
        int cur = half_edges_[he].next;
        int guard = 0;
        while (half_edges_[cur].twin >= 0) {
            cur = half_edges_[half_edges_[cur].twin].next;
            if (++guard > (int)half_edges_.size()) return -1; // safety
        }
        return cur;
    };

    int he  = start_he;
    int guard = 0;
    int max_iter = (int)half_edges_.size() + 4;
    do {
        cb(he);
        he = boundary_next(he);
        if (he < 0) break;
        if (++guard > max_iter) break;
    } while (he != start_he);
}

std::vector<int> HalfEdgeMesh::boundary_loop_vertices(int start_he) const {
    std::vector<int> verts;
    boundary_loop(start_he, [&](int he) {
        // The source vertex of boundary half-edge he is prev(he).vertex
        verts.push_back(half_edges_[half_edges_[he].prev].vertex);
    });
    return verts;
}

// -----------------------------------------------------------------------
// valence — number of incident faces (= degree for interior vertices).
// -----------------------------------------------------------------------
int HalfEdgeMesh::valence(int v) const {
    int count = 0;
    vertex_ring(v, [&](int /*he*/) { ++count; });
    return count;
}

void HalfEdgeMesh::face_ring(int f, std::function<void(int)> cb) const {
    int start = faces_[f].half_edge;
    int he = start;
    do { cb(he); he = half_edges_[he].next; } while (he != start);
}

std::array<int,3> HalfEdgeMesh::face_vertices(int f) const {
    int he = faces_[f].half_edge;
    return {
        half_edges_[half_edges_[he].prev].vertex,
        half_edges_[he].vertex,
        half_edges_[half_edges_[he].next].vertex
    };
}

std::array<int,3> HalfEdgeMesh::face_half_edges(int f) const {
    int he = faces_[f].half_edge;
    return {he, half_edges_[he].next, half_edges_[he].prev};
}

int HalfEdgeMesh::find_half_edge(int from, int to) const {
    int lo = std::min(from,to), hi = std::max(from,to);
    auto it = edge_map_.find({lo, hi});
    if (it == edge_map_.end()) return -1;
    for (int he : it->second) {
        // Check direction: half-edge points to 'to', comes from 'from'
        // The from vertex is the vertex at he.prev.vertex
        int prev = half_edges_[he].prev;
        if (half_edges_[prev].vertex == from && half_edges_[he].vertex == to)
            return he;
    }
    return -1;
}

bool HalfEdgeMesh::is_boundary_edge(int a, int b) const {
    int lo = std::min(a,b), hi = std::max(a,b);
    auto it = edge_map_.find({lo,hi});
    if (it == edge_map_.end()) return false;
    return it->second.size() == 1;
}

bool HalfEdgeMesh::is_boundary_vertex(int v) const {
    bool bv = false;
    vertex_ring(v, [&](int he){
        if (half_edges_[he].twin < 0) bv = true;
    });
    return bv;
}

// -----------------------------------------------------------------------
// Topology queries
// -----------------------------------------------------------------------

int HalfEdgeMesh::euler_characteristic() const {
    int V = (int)vertices_.size();
    int E = (int)edge_map_.size();
    int F = (int)faces_.size();
    return V - E + F;
}

// Public wrapper — see BUG FIX (v104) note in halfedge.h.
int HalfEdgeMesh::num_connected_components() const {
    return num_connected_components_internal();
}

int HalfEdgeMesh::genus() const {    // For a compact orientable surface with C connected components, b boundary
    // loops, and Euler characteristic χ:
    //
    //   χ = 2C − 2g − b   →   g = (2C − χ − b) / 2
    //
    // BUG FIX (v103): The previous formula  (2 − χ − b) / 2  assumed a SINGLE
    // connected component (C = 1).  After the repair pass an input mesh may
    // legitimately split into 2+ components (e.g. two separate shells).  Using
    // C = 1 in that case yields a negative or garbage genus value, which
    // corrupts the topology report and the downstream seam-cut budget.
    //
    // Fix: count connected components via a local BFS over the face-adjacency
    // graph (identical logic to topology.cpp::connected_components() but
    // inlined here so HalfEdgeMesh has no dependency on topology.h).
    int chi = euler_characteristic();
    int b   = num_boundary_loops();
    int C   = num_connected_components_internal();
    // Integer divide is exact: χ = 2C − 2g − b guarantees (2C − χ − b) is even
    // for any closed orientable surface.
    return (2 * C - chi - b) / 2;
}

int HalfEdgeMesh::num_connected_components_internal() const {
    const int nf = static_cast<int>(faces_.size());
    if (nf == 0) {
        // Pure point cloud — every vertex is its own component.
        return static_cast<int>(vertices_.size());
    }

    std::vector<bool> visited_face(nf, false);
    int comps = 0;

    for (int seed = 0; seed < nf; ++seed) {
        if (visited_face[seed]) continue;
        ++comps;
        std::queue<int> q;
        q.push(seed);
        visited_face[seed] = true;
        while (!q.empty()) {
            int fi = q.front(); q.pop();
            int he_start = faces_[fi].half_edge;
            int he       = he_start;
            do {
                int tw = half_edges_[he].twin;
                if (tw >= 0) {
                    int fj = half_edges_[tw].face;
                    if (fj >= 0 && !visited_face[fj]) {
                        visited_face[fj] = true;
                        q.push(fj);
                    }
                }
                he = half_edges_[he].next;
            } while (he != he_start);
        }
    }

    // Each isolated vertex (no incident face) is its own component.
    for (const Vertex& v : vertices_)
        if (v.half_edge < 0) ++comps;

    return comps;
}

int HalfEdgeMesh::num_boundary_loops() const {
    if (boundary_edges_.empty()) return 0;
    // BFS over boundary edges to count connected loops
    std::unordered_map<int, std::vector<int>> bnd_adj;
    for (auto& [lo, hi] : boundary_edges_) {
        bnd_adj[lo].push_back(hi);
        bnd_adj[hi].push_back(lo);
    }
    std::unordered_map<int, bool> visited;
    int loops = 0;
    for (auto& [v, _] : bnd_adj) {
        if (visited.count(v)) continue;
        std::queue<int> q;
        q.push(v);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            if (visited[u]) continue;
            visited[u] = true;
            for (int nb : bnd_adj[u])
                if (!visited.count(nb)) q.push(nb);
        }
        ++loops;
    }
    return loops;
}

// -----------------------------------------------------------------------
// Geometry helpers
// -----------------------------------------------------------------------

Vec3 HalfEdgeMesh::face_normal(int f) const {
    auto [v0, v1, v2] = face_vertices(f);
    Vec3 a = vertices_[v1].pos - vertices_[v0].pos;
    Vec3 b = vertices_[v2].pos - vertices_[v0].pos;
    return a.cross(b);
}

double HalfEdgeMesh::face_area(int f) const {
    return 0.5 * face_normal(f).norm();
}

double HalfEdgeMesh::dihedral_angle(int he) const {
    int tw = half_edges_[he].twin;
    if (tw < 0) return 0.0;
    Vec3 n1_raw = face_normal(half_edges_[he].face);
    Vec3 n2_raw = face_normal(half_edges_[tw].face);
    double l1 = n1_raw.norm();
    double l2 = n2_raw.norm();
    // Guard against degenerate (zero-area) faces: return 0 rather than NaN.
    if (l1 < 1e-15 || l2 < 1e-15) return 0.0;
    Vec3 n1 = n1_raw / l1;
    Vec3 n2 = n2_raw / l2;
    double c = std::clamp(n1.dot(n2), -1.0, 1.0);
    return std::acos(c);
}

double HalfEdgeMesh::total_area() const {
    double total = 0;
    for (int fi = 0; fi < (int)faces_.size(); ++fi)
        total += face_area(fi);
    return total;
}

} // namespace qf
