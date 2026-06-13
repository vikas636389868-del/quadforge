#pragma once
#ifndef QUADFORGE_ENGINE_H
#define QUADFORGE_ENGINE_H

#include "types.h"
#include "api.h"
#include "mesh/halfedge.h"
#include "mesh/features.h"
// INTEGRATION (v101): mesh/symmetry.h provides SymmetryResult — the output of
// the reflective symmetry detection pass (roadmap §3.1 Step 1.4).  Included
// here so QFEngine can store the three per-axis results as a member and make
// them available to stage2 (field constraints) and stage3 (parametrization
// mirroring) without requiring those stages to re-detect symmetry.
#include "mesh/symmetry.h"
#include "field/curvature.h"
#include "field/cross_field.h"
#include "field/singularity.h"
#include "param/miq.h"
// INTEGRATION FIX (v84): param/igm.h was included directly in engine.cpp but
// was absent from engine.h.  engine.h is the canonical dependency manifest for
// the QFEngine class; any translation unit that includes engine.h to inspect
// the pipeline types must also be able to reach the IGM dispatch path.
// Adding it here makes the dependency explicit and prevents silent breakage if
// engine.cpp is ever split or if a caller tries to use IGM types via engine.h.
#include "param/igm.h"
#include <string>
#include <atomic>
#include <memory>
#include <functional>
#include <vector>
#include <cstdint>

namespace qf {

/**
 * Feature detection configuration flags.
 */
struct FeatureDetectionFlags {
    bool   auto_detect_hard_edges = true;
    double hard_edge_angle_deg    = 30.0;
    bool   use_normals            = false;
    bool   use_materials          = false;
    bool   use_uv_seams           = false;
};

/**
 * QFEngine — pipeline orchestrator.
 *
 * Owns all intermediate state for one remesh operation and provides
 * the six-stage pipeline as named methods.  Instantiated once per
 * qf_remesh() call; not reused across calls.
 */
class QFEngine {
public:
    explicit QFEngine(const QFInputMesh* input,
                      const QFParams*    params,
                      QFProgressCallback cb,
                      void*              user_data);
    ~QFEngine();

    // Disable copy; pipeline state is not copyable
    QFEngine(const QFEngine&)            = delete;
    QFEngine& operator=(const QFEngine&) = delete;

    /**
     * Run the full six-stage pipeline.
     * Returns a heap-allocated QFResult that the caller must free with
     * qf_free_result().  Returns nullptr on failure (check qf_last_error()).
     */
    QFResult* run();

    // Abort flag — set from progress callback returning non-zero
    std::atomic<bool> abort_flag{false};

private:
    // ---------- stage methods ----------
    bool stage1_preprocess();
    bool stage2_cross_field();
    bool stage3_parametrize();
    bool stage4_extract();
    bool stage5_postprocess();
    bool stage6_output(QFResult* result);

    // ---------- helpers ----------
    void report_progress(int stage, float progress, const char* name);
    void set_error(const std::string& msg);

    // ---------- input / params ----------
    const QFInputMesh* m_input;
    const QFParams*    m_params;
    QFProgressCallback m_cb;
    void*              m_user_data;

    // Raw input arrays (views into m_input — zero-copy)
    const float*   m_pos;
    const int32_t* m_faces_flat;
    const int32_t* m_face_sizes;
    int32_t        m_nv;
    int32_t        m_nf;
    int32_t        m_nt;
    float          m_base_edge_len;

    // ---------- pipeline state (populated by stages) ----------
    std::unique_ptr<qf::HalfEdgeMesh>          m_mesh;
    qf::FeatureData                            m_features;
    qf::CurvatureData                          m_curvature;
    Eigen::VectorXd                            m_sizing;
    qf::CrossField                             m_field;
    std::vector<qf::SingularityInfo>           m_singularities;
    qf::UVParam                                m_uv;
    qf::QuadMesh                               m_quad_mesh;

    // Per-axis reflective symmetry analysis results (roadmap §3.1 Step 1.4).
    // Populated in stage1_preprocess() by detect_all_symmetry_axes().
    // Index 0=X, 1=Y, 2=Z matching SymmetryAxis enum values.
    // Consumed by stage2_cross_field() (field constraints) and stage3
    // (parametrization mirroring).
    std::array<qf::SymmetryResult, 3>          m_symmetry;

    // BUG FIX (v84 / B3): m_poly_to_tri maps each original input face index to
    // the index of its FIRST triangulated triangle in m_mesh.  Populated in
    // stage1_preprocess() by triangulate_input().  Required by stage6_output()
    // to build the tri→original-face reverse map for material ID transfer.
    // Without this member, material_ids were always all-zeros in QFResult even
    // when the input mesh had per-face materials — the pack_result() comment
    // "engine.cpp fills these after the call" was a permanent lie.
    std::vector<int32_t> m_poly_to_tri;

    // ---------- error state ----------
    std::string m_last_error;
};

} // namespace qf

#endif // QUADFORGE_ENGINE_H
