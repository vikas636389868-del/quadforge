#pragma once
/* quadforge/mesh/halfedge.h — Half-edge mesh data structure.
 *
 * Supports manifold triangle meshes with arbitrary genus, boundaries,
 * and basic non-manifold edge detection.  Construction from flat arrays
 * in O(V + F) time.
 */

#ifndef QUADFORGE_MESH_HALFEDGE_H
#define QUADFORGE_MESH_HALFEDGE_H

#include <array>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <functional>

#include "../types.h"

namespace qf {

/** Single half-edge record. */
struct HalfEdge {
    int vertex{-1};   // destination vertex
    int face{-1};     // owning face (-1 = boundary)
    int next{-1};     // next half-edge in face loop
    int prev{-1};     // previous half-edge in face loop
    int twin{-1};     // opposite half-edge (-1 = boundary)
};

/** Per-vertex record. */
struct Vertex {
    Vec3 pos;
    int  half_edge{-1};  // one outgoing half-edge
};

/** Per-face record (triangle). */
struct Face {
    int half_edge{-1};   // one half-edge in this face
};

/**
 * Half-edge mesh.
 *
 * Terminology:
 *   vertex v owns the half-edge array entry where he.vertex == v.
 *   A boundary half-edge has face == -1 and twin == -1.
 */
class HalfEdgeMesh {
public:
    // ---------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------
    HalfEdgeMesh() = default;
    /**
     * Build from flat arrays.
     * @param positions  [V*3] vertex XYZ (row-major)
     * @param tri_faces  [F*3] triangle vertex indices
     */
    HalfEdgeMesh(const std::vector<Vec3>& positions,
                 const std::vector<std::array<int,3>>& tri_faces);

    /** Convenience: build from Eigen matrices. */
    static HalfEdgeMesh from_matrices(
        const Eigen::MatrixXd& V,   // V×3
        const Eigen::MatrixXi& F    // F×3
    );

    // ---------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------

    int num_vertices()  const { return static_cast<int>(vertices_.size());  }
    int num_faces()     const { return static_cast<int>(faces_.size());     }
    int num_half_edges()const { return static_cast<int>(half_edges_.size());}

    const Vertex&   vertex(int i)    const { return vertices_[i];    }
    const Face&     face(int i)      const { return faces_[i];       }
    const HalfEdge& half_edge(int i) const { return half_edges_[i];  }

    Vec3  vertex_pos(int i) const { return vertices_[i].pos; }

    /** True if edge (a,b) is a boundary edge. */
    bool is_boundary_edge(int a, int b) const;

    /** True if vertex v is on the mesh boundary. */
    bool is_boundary_vertex(int v) const;

    // ---------------------------------------------------------------
    // Traversal
    // ---------------------------------------------------------------

    /** Visit all half-edges around vertex v's outgoing ring. */
    void vertex_ring(int v, std::function<void(int he)> callback) const;

    /** Visit all vertices adjacent to v. */
    void vertex_neighbours(int v, std::function<void(int nv)> callback) const;

    /** Visit all faces adjacent to v. */
    void vertex_faces(int v, std::function<void(int fi)> callback) const;

    /**
     * Visit each half-edge emanating from vertex v in the 'edge ring' sense —
     * i.e. every (v, w) directed edge leaving v, identical to vertex_ring but
     * named to match the roadmap traversal vocabulary.
     */
    void edge_ring(int v, std::function<void(int he)> callback) const;

    /**
     * Walk a boundary loop starting from boundary half-edge start_he and
     * invoke callback for each boundary half-edge in order.
     * start_he must satisfy half_edge(start_he).twin == -1.
     * The walk is guaranteed to terminate (boundary loops are finite).
     */
    void boundary_loop(int start_he, std::function<void(int he)> callback) const;

    /**
     * Return the ordered vertex indices of the boundary loop that
     * contains half-edge start_he.  start_he must be a boundary half-edge.
     */
    std::vector<int> boundary_loop_vertices(int start_he) const;

    /** Return the number of faces incident on vertex v (its valence). */
    int valence(int v) const;

    /** Visit all half-edges on face f's boundary. */
    void face_ring(int f, std::function<void(int he)> callback) const;

    /** Get the 3 vertex indices of triangle f. */
    std::array<int,3> face_vertices(int f) const;

    /** Get the 3 half-edge indices of triangle f. */
    std::array<int,3> face_half_edges(int f) const;

    /** Get the half-edge index for directed edge (a→b), or -1. */
    int find_half_edge(int from, int to) const;

    // ---------------------------------------------------------------
    // Topology queries
    // ---------------------------------------------------------------

    bool is_manifold()       const { return non_manifold_edges_.empty(); }
    bool is_closed()         const { return boundary_edges_.empty(); }
    int  euler_characteristic() const;
    int  genus()             const;
    int  num_boundary_loops()   const;

    /**
     * Return the number of connected components in this mesh.
     *
     * BUG FIX (v104): num_connected_components_internal() was private, so no
     * external module (especially compute_mesh_stats() in mesh_stats.cpp) could
     * query the component count.  compute_mesh_stats() left
     * ComplexityIndicators::num_connected_components permanently at 0, corrupting
     * the mesh-difficulty score, adaptive-pipeline classifier, and CSV output.
     *
     * This thin public wrapper delegates to the existing private BFS
     * implementation without duplicating any logic.
     */
    int  num_connected_components() const;

    const std::vector<std::pair<int,int>>& non_manifold_edges() const { return non_manifold_edges_; }
    const std::vector<std::pair<int,int>>& boundary_edges()     const { return boundary_edges_; }

    // ---------------------------------------------------------------
    // Geometry helpers
    // ---------------------------------------------------------------

    /** Face normal (unnormalized). */
    Vec3  face_normal(int f) const;

    /** Face area. */
    double face_area(int f) const;

    /** Dihedral angle between faces sharing directed half-edge he (radians). */
    double dihedral_angle(int he) const;

    /** Total surface area. */
    double total_area() const;

    // ---------------------------------------------------------------
    // Data arrays (for read-only access by other engine modules)
    // ---------------------------------------------------------------
    const std::vector<Vertex>&   vertices()   const { return vertices_;   }
    const std::vector<Face>&     faces()      const { return faces_;      }
    const std::vector<HalfEdge>& half_edges() const { return half_edges_; }

private:
    std::vector<Vertex>   vertices_;
    std::vector<Face>     faces_;
    std::vector<HalfEdge> half_edges_;

    // Edge pair map: (v_lo, v_hi) → [he1, he2, ...]
    using EdgeKey = std::pair<int,int>;
    struct EdgeKeyHash {
        size_t operator()(const EdgeKey& k) const {
            // BUG FIX (v94): Previous code was
            //   static_cast<int64_t>(k.first) << 32
            // which is undefined behavior when k.first is negative (left-shift
            // of a negative signed integer is UB in C++ [expr.shift §7.6.7]).
            // Although vertex indices are logically non-negative, the UB can be
            // triggered by defensive code paths that temporarily store sentinel
            // values (-1) in half-edge vertex fields.
            //
            // Fix: cast both components through uint32_t before promoting to
            // int64_t / uint64_t, exactly as seam_utils.h::seam_edge_key() does.
            // uint32_t → uint64_t shift is well-defined for all shift amounts in
            // [0, 31].  The final hash value is unchanged for non-negative indices
            // (the common case), so existing hash-map data is not invalidated.
            return std::hash<int64_t>()(
                (static_cast<int64_t>(static_cast<uint32_t>(k.first)) << 32)
                | static_cast<int64_t>(static_cast<uint32_t>(k.second)));
        }
    };
    std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash> edge_map_;

    std::vector<std::pair<int,int>> non_manifold_edges_;
    std::vector<std::pair<int,int>> boundary_edges_;

    void build_connectivity();
    void detect_topology_issues();
    // BUG FIX (v103): used by genus() to handle multi-component meshes correctly.
    int  num_connected_components_internal() const;
};

} // namespace qf

#endif // QUADFORGE_MESH_HALFEDGE_H
