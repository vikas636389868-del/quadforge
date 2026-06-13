#pragma once
/* quadforge/extract/patch_layout.h — Motorcycle graph surface decomposition.
 *
 * ROLE IN THE PIPELINE
 * --------------------
 * extraction_method = 1 (MotorcycleGraph) uses a fundamentally different
 * strategy from iso-line extraction (method 0) and dual contouring (method 2):
 *
 *   • Methods 0 and 2 trace the entire integer UV grid globally, then resolve
 *     T-junctions after the fact.
 *
 *   • Method 1 FIRST partitions the surface into rectangular "patches" using
 *     the motorcycle graph (Eppstein 2008), THEN parameterises each patch
 *     independently with a simple Tutte/Floater mapping, and FINALLY runs a
 *     per-patch isoline extraction.  Because each patch is topologically a
 *     disk with four boundary corners, the parametrization is bijective and
 *     T-junctions cannot arise inside a patch — only at patch boundaries,
 *     where they are resolved by construction.
 *
 * The advantage of method 1 is robustness: it never produces T-junctions in
 * the interior of a patch, and it scales well to meshes with many singularities
 * because each patch parametrization is a small, independent linear solve.
 *
 * DATA STRUCTURES
 * ---------------
 * The motorcycle graph is a planar graph G = (V_m, E_m) embedded on the mesh:
 *
 *   PatchVertex  — a node in G.  Corresponds to a singularity, boundary
 *                  corner, or boundary-curve junction on the original mesh.
 *
 *   PatchEdge    — a directed edge in G.  Corresponds to a chain of mesh
 *                  triangles (a "motorcycle track") connecting two PatchVertices.
 *                  Each track is a monotone path in the UV domain.
 *
 *   PatchFace    — a rectangular region bounded by four PatchEdges, forming
 *                  a single patch for independent parametrization.  Degenerate
 *                  patches (e.g., triangular) are supported via n_corners < 4.
 *
 * ALGORITHM OUTLINE (build_patch_layout)
 * ----------------------------------------
 *  1. Seed one "rider" per singularity (+½ index → 1 rider, −½ index → 3,
 *     matching the MIQ convention for 4-RoSy fields).  Boundary loops seed
 *     additional riders at every boundary corner.
 *
 *  2. Advance each rider along the dominant field direction (stored in
 *     CrossField::face_field) by crossing triangle edges.  A rider halts when
 *     it hits another rider's track, a singularity, or the mesh boundary.
 *
 *  3. The set of all halted riders' tracks forms the motorcycle graph edges.
 *
 *  4. Faces (patches) are extracted from the planar graph by traversing the
 *     half-edge structure of G.
 *
 * REFERENCES
 * ----------
 * Eppstein et al. (2008) "Motorcycle Graphs: Canonical Quad Mesh Partitioning"
 *   — original algorithm definition.
 * Bommes et al. (2013) "Integer-Grid Maps for Reliable Quad Meshing," §3
 *   — how the motorcycle graph fits into the MIQ pipeline.
 * Campen et al. (2012) "Dual Loops Meshing: Quality Quad Layouts on Manifolds"
 *   — per-patch Tutte parametrization strategy.
 */

#ifndef QUADFORGE_EXTRACT_PATCH_LAYOUT_H
#define QUADFORGE_EXTRACT_PATCH_LAYOUT_H

#include "../types.h"
#include "../mesh/halfedge.h"
#include "../field/singularity.h"
#include "../field/cross_field.h"

#include <cstdint>
#include <vector>
#include <array>
#include <unordered_map>

namespace qf {

/* ======================================================================
 * PatchVertex — node in the motorcycle graph
 * ====================================================================== */

/**
 * Classification of a PatchVertex (determines how many motorcycle tracks
 * emanate from it and whether it forces a specific valence in the output).
 */
enum class PatchVertexKind : uint8_t {
    Singularity     = 0,  ///< Interior singularity (index ±¼ → 1 or 3 tracks)
    BoundaryCorner  = 1,  ///< Convex boundary corner (1 track)
    BoundaryJunction= 2,  ///< Concave/T-junction on boundary (≥2 tracks)
    ArtificialSeed  = 3,  ///< Injected to limit patch size
};

/** A node in the motorcycle graph. */
struct PatchVertex {
    int              mesh_vertex;   ///< Corresponding vertex index in HalfEdgeMesh
    PatchVertexKind  kind;
    Vec3             position;      ///< 3D world position
    double           uv[2];         ///< UV coordinates at this vertex
    int              num_tracks;    ///< Number of outgoing motorcycle tracks
};


/* ======================================================================
 * PatchEdge — directed motorcycle track
 * ====================================================================== */

/**
 * A motorcycle track connecting two PatchVertex nodes.
 *
 * track_verts contains the ordered list of mesh vertex indices traversed
 * along the track (including source and target PatchVertex mesh_vertex
 * at indices 0 and back() respectively).
 *
 * track_faces contains the face indices of the triangles whose edge the
 * rider crossed — one fewer element than track_verts.
 */
struct PatchEdge {
    int source;   ///< Index into PatchLayout::vertices (origin PatchVertex)
    int target;   ///< Index into PatchLayout::vertices (destination PatchVertex)
    int twin;     ///< Index of the reverse-direction PatchEdge (-1 if boundary)

    std::vector<int> track_verts;   ///< Mesh vertices along the track
    std::vector<int> track_faces;   ///< Mesh faces traversed

    /** Approximate 3D arc-length of the track. */
    double arc_length{0.0};

    /** Number of integer UV steps this track spans (≥1). */
    int    uv_steps{1};

    /** True if this track lies on the mesh boundary. */
    bool   on_boundary{false};
};


/* ======================================================================
 * PatchFace — a single rectangular patch
 * ====================================================================== */

/**
 * A rectangular surface patch bounded by motorcycle tracks.
 *
 * Most patches have exactly 4 corners (n_corners==4).  Degenerate patches
 * near complex singularity clusters may have 3 corners (triangle patch)
 * or 5+ corners (pentagon patch) — these require special handling during
 * per-patch parametrization.
 *
 * corners[i] and edges[i] are ordered counter-clockwise:
 *   edge[0] runs from corners[0] → corners[1],
 *   edge[1] runs from corners[1] → corners[2], etc.
 */
struct PatchFace {
    static constexpr int MAX_CORNERS = 8;  ///< Safety cap

    // BUG FIX (v91): n_corners, corners[], and edges[] had no default
    // member initializers.  A default-constructed PatchFace (e.g. created
    // by std::vector<PatchFace>::resize()) left n_corners as an indeterminate
    // value, causing is_degenerate() to return undefined results and
    // downstream loops over corners[0..n_corners-1] to read garbage.
    // Fix: add brace-initializers so all three members are zero-initialized.
    int n_corners{0};                    ///< Number of corners (usually 4)
    int corners[MAX_CORNERS]{};          ///< PatchVertex indices (CCW order)
    int edges  [MAX_CORNERS]{};          ///< PatchEdge indices for each side

    /** Flat list of all mesh face indices inside this patch. */
    std::vector<int> contained_faces;

    /** Target quad resolution for this patch (quads per shortest side). */
    int target_quads_u{4};
    int target_quads_v{4};

    /** True if the patch is degenerate (n_corners != 4). */
    bool is_degenerate() const { return n_corners != 4; }
};


/* ======================================================================
 * PatchLayout — the full motorcycle graph
 * ====================================================================== */

/**
 * Complete motorcycle graph decomposition of the mesh surface.
 *
 * Invariants after a successful build_patch_layout():
 *   • Every mesh triangle is contained in exactly one PatchFace.
 *   • Every PatchFace edge is a PatchEdge (with a valid twin except
 *     for edges on the mesh boundary).
 *   • PatchVertex::mesh_vertex is a valid vertex index in [0, nv).
 */
struct PatchLayout {
    std::vector<PatchVertex>  vertices;   ///< Nodes (singularities, corners)
    std::vector<PatchEdge>    edges;      ///< Directed motorcycle tracks
    std::vector<PatchFace>    faces;      ///< Rectangular patches

    /** Statistics filled by build_patch_layout(). */
    int  n_quads_total{0};     ///< Sum of (u_steps * v_steps) over all patches
    int  n_degenerate{0};      ///< Number of non-4-corner patches
    double mean_aspect{1.0};   ///< Mean patch aspect ratio (u_steps / v_steps).
                               ///< 1.0 = perfectly square patches; > 1 = elongated.
                               ///< Default is 1.0 (the identity/neutral value).
                               ///< A value of 0.0 is geometrically impossible and
                               ///< indicates build_patch_layout() was never called.
};


/* ======================================================================
 * build_patch_layout — main entry point
 * ====================================================================== */

/**
 * Build the motorcycle graph patch decomposition from the cross-field
 * and parametrization results.
 *
 * Prerequisites:
 *   • mesh is a valid manifold triangle mesh.
 *   • field contains a computed CrossField (face_field is non-empty).
 *   • uv contains a valid UVParam (U, V sized to nv, seam_edges set).
 *   • singularities contains detected SingularityInfo records.
 *
 * The function runs the motorcycle graph algorithm:
 *  1. Seeds riders at singularities and boundary corners.
 *  2. Advances each rider across triangle edges following the field
 *     direction (greatest U or V gradient component).
 *  3. Halts riders on collision.
 *  4. Extracts PatchFace regions from the resulting track network.
 *  5. Computes target quad resolution per patch from the global
 *     target_quad_count (estimated via patch area fraction).
 *
 * @param mesh              Input triangle mesh
 * @param field             Computed 4-RoSy cross-field
 * @param uv                UV parametrization with seam_edges
 * @param singularities     Detected and (optionally) relocated singularities
 * @param target_quad_count Target total quad count from QFParams
 * @param num_threads       Thread count for parallel rider advancement (0=auto)
 * @return                  PatchLayout (empty on failure)
 */
PatchLayout build_patch_layout(
    const HalfEdgeMesh&                     mesh,
    const CrossField&                       field,
    const UVParam&                          uv,
    const std::vector<SingularityInfo>&     singularities,
    int                                     target_quad_count = 5000,
    int                                     num_threads       = 0);


/* ======================================================================
 * extract_quads_motorcycle — full method=1 pipeline entry point
 * ====================================================================== */

/**
 * Extract a quad mesh using the motorcycle graph method (extraction_method=1).
 *
 * This is the top-level function called by engine.cpp for method=1.  It
 * internally calls build_patch_layout(), then for each patch:
 *
 *   1. Extracts the sub-mesh triangles contained in the patch.
 *   2. Runs a Tutte boundary-mapped parametrization (square boundary,
 *      uniform weights) to obtain a [0,1]² UV for the patch.
 *   3. Scales the UV to (target_quads_u × target_quads_v) and calls
 *      extract_quads_from_isolines() on the sub-mesh.
 *   4. Appends the per-patch quad mesh to the global result, mapping
 *      local vertex indices back to global mesh coordinates via BVH
 *      reprojection.
 *
 * Boundary edges shared between adjacent patches are merged (vertex
 * welding with tolerance 1e-6) to produce a watertight global quad mesh
 * with no T-junctions.
 *
 * @param mesh              Input triangle mesh
 * @param field             Computed cross-field (needed for rider seeding)
 * @param uv                UV parametrization from the param stage
 * @param singularities     Detected singularities
 * @param target_quad_count Global target (distributed across patches)
 * @param num_threads       Thread count (0 = auto)
 * @return                  Quad mesh (T-junction-free by construction)
 */
QuadMesh extract_quads_motorcycle(
    const HalfEdgeMesh&                     mesh,
    const CrossField&                       field,
    const UVParam&                          uv,
    const std::vector<SingularityInfo>&     singularities,
    int                                     target_quad_count = 5000,
    int                                     num_threads       = 0);


/* ======================================================================
 * Utility: patch quality metrics
 * ====================================================================== */

/**
 * Compute the aspect ratio of a PatchFace: longest_side / shortest_side.
 *
 * Uses 3D arc-lengths of the bounding PatchEdges.
 * Returns 1.0 for a perfectly square patch, > 1 for elongated patches.
 *
 * DEGENERATE PATCH WARNING: If any bounding PatchEdge has arc_length == 0
 * (e.g., a collapsed edge from an overlapping singularity), the result is
 * +infinity (shortest_side = 0).  Callers MUST filter degenerate patches
 * (is_degenerate() == true, or n_corners < 2) before calling this function,
 * and should guard against +Inf by checking arc_length > 0 on all edges.
 *
 * @param layout  PatchLayout containing the face and edge data
 * @param fi      Index of the PatchFace to evaluate
 * @return        Aspect ratio in [1, ∞), or +Inf for degenerate patches
 */
double patch_aspect_ratio(const PatchLayout& layout, int fi);

/**
 * Estimate the number of quads that should be allocated to patch fi,
 * given a global target and the patch's area fraction.
 *
 * Result is clamped to [1, max_quads_per_patch].
 *
 * @param layout              Full PatchLayout (needed for patch areas)
 * @param fi                  Patch index
 * @param total_mesh_area     Precomputed total mesh surface area
 * @param global_target       QFParams::target_quad_count
 * @param max_quads_per_patch Safety cap (default 10000)
 * @return                    Estimated quad count for this patch
 */
int patch_quad_budget(
    const PatchLayout& layout,
    int                fi,
    double             total_mesh_area,
    int                global_target,
    int                max_quads_per_patch = 10000);

} // namespace qf

#endif // QUADFORGE_EXTRACT_PATCH_LAYOUT_H
