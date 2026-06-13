/**
 * engine.cpp — QFEngine: six-stage pipeline orchestrator.
 */

#include "../include/quadforge/engine.h"
#include "../include/quadforge/types.h"
#include "../include/quadforge/mesh/halfedge.h"
#include "../include/quadforge/mesh/repair.h"
#include "../include/quadforge/mesh/topology.h"
#include "../include/quadforge/mesh/spatial.h"
#include "../include/quadforge/mesh/features.h"
// INTEGRATION (v101): mesh/symmetry.h provides detect_all_symmetry_axes(),
// detect_reflective_symmetry(), and related utilities.  Included here so
// stage1_preprocess() can run the roadmap §3.1 Step 1.4 symmetry detection
// pass and store results in m_symmetry for use by stage2 and stage3.
#include "../include/quadforge/mesh/symmetry.h"
// FIX (v100-B4): mesh/sizing_field.h was never included from engine.cpp.
// The mesh-layer sizing utilities (apply_curvature_modifier,
// apply_vertex_color_modifier, apply_feature_proximity_modifier,
// curvature_magnitudes_from_data) were compiled but completely invisible to
// the engine translation unit.  Adding the direct include here makes the
// mesh-layer API visible to engine.cpp and ensures future pipeline stages
// (e.g. a per-stage sizing refresh in stage2_cross_field) can call the
// canonical mesh-layer functions without re-introducing duplication.
#include "../include/quadforge/mesh/sizing_field.h"
#include "../include/quadforge/field/curvature.h"
#include "../include/quadforge/field/cross_field.h"
#include "../include/quadforge/field/singularity.h"
#include "../include/quadforge/field/smoothing.h"   // FIX (v54+): field smoothing pass
// BUG FIX (v83): constraints.h was not directly included in engine.cpp.
// build_edge_to_faces() and build_feature_edge_constraints() (both declared in
// constraints.h) were only reachable via the transitive include path:
//   engine.cpp → field/smoothing.h → field/constraints.h
// This is fragile: if smoothing.h ever stops including constraints.h
// (a legitimate refactor), engine.cpp silently stops seeing those two
// declarations and the build breaks.  Adding the direct include here makes
// the dependency explicit and stable.
#include "../include/quadforge/field/constraints.h"
#include "../include/quadforge/param/metric.h"
#include "../include/quadforge/param/combing.h"
#include "../include/quadforge/param/seams.h"   // FIX BUG C (v72): was missing
#include "../include/quadforge/param/miq.h"
#include "../include/quadforge/param/igm.h"
#include "../include/quadforge/extract/isolines.h"
#include "../include/quadforge/extract/motorcycle.h"
#include "../include/quadforge/extract/cleanup.h"
#include "../include/quadforge/extract/quad_mesh.h"
#include "../include/quadforge/extract/dual_contour.h"
#include "../include/quadforge/extract/edge_flow_refine.h"
#include "../include/quadforge/extract/patch_layout.h"
#include "../include/quadforge/extract/seam_utils.h"
#include "../include/quadforge/postprocess/smooth.h"
#include "../include/quadforge/postprocess/project.h"
#include "../include/quadforge/postprocess/snap.h"
#include "../include/quadforge/postprocess/subdiv.h"
#include "../include/quadforge/postprocess/metrics.h"
#include "../include/quadforge/accel/omp_utils.h"
// INTEGRATION FIX: thread_pool.h was fully implemented (LIFO queue,
// submit(), parallel_for(), parallel_reduce(), global_thread_pool()) but
// never #include'd from any engine file — it was completely dead code.
// Including it here makes global_thread_pool() available to the engine
// orchestrator so future pipeline stages can submit tasks to the persistent
// pool instead of spawning transient OpenMP regions for every stage.
// The pool is initialised lazily on first call to global_thread_pool().
#include "../include/quadforge/accel/thread_pool.h"
// INTEGRATION FIX (v82): mesh_io.h was never included from engine.cpp, so
// triangulate_input() (better quad-diagonal split), compute_face_normals(),
// compute_vertex_normals() (angle-weighted), and pack_result() (diagonal-
// cross quad normals + correct material_ids zero-init) were compiled but
// completely dead.  Stage 1 now delegates triangulation to triangulate_input()
// and Stage 6 delegates result packing to pack_result().
// NOTE (v84): mesh/spatial.h is already included at line 10 above so
// TriangleBVH is available for stage6_output()'s material-transfer BVH
// construction — no second include needed.
#include "../include/quadforge/mesh/mesh_io.h"

#include <cstring>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <array>
#include <chrono>

// Private cross-TU declaration (see src/_internal.h)
#include "_internal.h"

namespace qf {

// -----------------------------------------------------------------------
// Helpers: convert flat C arrays → Eigen / internal types
// -----------------------------------------------------------------------

static std::vector<Vec3> positions_from_c(const float* pos, int32_t nv) {
    std::vector<Vec3> out(nv);
    for (int i = 0; i < nv; ++i)
        out[i] = Vec3(pos[3*i], pos[3*i+1], pos[3*i+2]);
    return out;
}

// NOTE (v82): faces_from_c() has been removed.  It used a simple fan
// triangulation for quads that always split v0-v1-v2 / v0-v2-v3 regardless
// of diagonal lengths, producing degenerate slivers on non-planar quads.
// mesh_io::triangulate_input() replaces it: it picks the shorter diagonal for
// quads and correctly tracks poly_to_tri for material transfer.

// -----------------------------------------------------------------------
// QFEngine constructor
// -----------------------------------------------------------------------

QFEngine::QFEngine(const QFInputMesh* input, const QFParams* params,
                   QFProgressCallback cb, void* user_data)
    : m_input(input), m_params(params), m_cb(cb), m_user_data(user_data)
    , m_pos(input->positions), m_faces_flat(input->faces)
    , m_face_sizes(input->face_sizes)
    , m_nv(input->num_vertices), m_nf(input->num_faces)
    , m_nt(0), m_base_edge_len(0.f)
{}

QFEngine::~QFEngine() = default;

void QFEngine::report_progress(int stage, float progress, const char* name) {
    if (m_cb) {
        int ret = m_cb(stage, progress, name, m_user_data);
        if (ret != 0) abort_flag.store(true);
    }
}

void QFEngine::set_error(const std::string& msg) {
    m_last_error = msg;
    _qf_set_global_error(msg);
}

// -----------------------------------------------------------------------
// run()
// -----------------------------------------------------------------------

QFResult* QFEngine::run() {
    // BUG FIX v29: record wall-clock start so elapsed_seconds is populated in
    // the result.  Previously the field was allocated by `new QFResult{}` and
    // never written, so it was always 0.0 regardless of actual runtime.
    using Clock = std::chrono::steady_clock;
    auto t_start = Clock::now();

    if (!stage1_preprocess()) return nullptr;
    if (abort_flag.load()) { set_error("Aborted"); return nullptr; }
    if (!stage2_cross_field()) return nullptr;
    if (abort_flag.load()) { set_error("Aborted"); return nullptr; }
    if (!stage3_parametrize()) return nullptr;
    if (abort_flag.load()) { set_error("Aborted"); return nullptr; }
    if (!stage4_extract()) return nullptr;
    if (abort_flag.load()) { set_error("Aborted"); return nullptr; }
    if (!stage5_postprocess()) return nullptr;
    if (abort_flag.load()) { set_error("Aborted"); return nullptr; }

    QFResult* result = new QFResult{};
    if (!stage6_output(result)) { delete result; return nullptr; }

    // Populate elapsed_seconds now that all stages have finished.
    auto t_end = Clock::now();
    result->elapsed_seconds = static_cast<float>(
        std::chrono::duration<double>(t_end - t_start).count());

    return result;
}

// -----------------------------------------------------------------------
// Stage 1: Preprocessing
// -----------------------------------------------------------------------

bool QFEngine::stage1_preprocess() {
    report_progress(0, 0.0f, "Preprocessing");

    // Build HalfEdgeMesh
    auto positions = positions_from_c(m_pos, m_nv);

    // INTEGRATION FIX (v82): Use triangulate_input() from mesh_io.cpp instead
    // of the old inline faces_from_c().  triangulate_input() uses the shorter
    // diagonal split for quads (better triangle quality → better cotangent
    // weights in the connection Laplacian) and correctly handles N-gons via
    // fan triangulation from vertex 0 — the same code path used by mesh_io
    // everywhere else in the engine so results are consistent.
    std::vector<int32_t> tris_flat;
    // BUG FIX (v84 / B3): Store the face→triangle-start map in the member
    // variable m_poly_to_tri so that stage6_output() can build the reverse
    // tri→original-face map needed for material ID transfer.  Previously this
    // was a local-only variable that was destroyed at the end of stage1,
    // making material ID transfer permanently impossible even though the
    // comment in pack_result() said "engine.cpp fills these after the call."
    int32_t tri_count = 0;
    triangulate_input(m_pos, m_faces_flat, m_face_sizes, m_nv, m_nf,
                      tris_flat, tri_count, m_poly_to_tri);
    m_nt = tri_count;

    if (m_nt == 0) { set_error("Stage 1: No triangles"); return false; }

    // Convert flat tris to std::array<int,3> for HalfEdgeMesh constructor
    std::vector<std::array<int,3>> tris(m_nt);
    for (int i = 0; i < m_nt; ++i)
        tris[i] = { tris_flat[i*3], tris_flat[i*3+1], tris_flat[i*3+2] };

    m_mesh = std::make_unique<HalfEdgeMesh>(positions, tris);
    report_progress(0, 0.3f, "Preprocessing");

    // ------------------------------------------------------------------
    // Pre-flight repair pass (Roadmap §11.2 / §13.7)
    // ------------------------------------------------------------------
    {
        MeshRepairOptions ropts;
        ropts.adaptive_weld        = true;
        ropts.repair_winding       = true;
        ropts.split_nm_vertices    = true;
        ropts.detect_self_intersect  = true;
        ropts.isolate_self_intersect = true;
        RepairReport rr = repair_mesh(*m_mesh, ropts);
        (void)rr;
    }
    report_progress(0, 0.35f, "Repairing mesh");

    // Topology report
    auto topo = analyse_topology(*m_mesh);
    (void)topo; // printed in debug mode

    // Feature detection
    FeatureDetectionFlags ff;
    ff.auto_detect_hard_edges = (m_params->auto_detect_hard_edges != 0);
    ff.hard_edge_angle_deg    = m_params->hard_edge_angle_deg;
    ff.use_normals            = (m_params->use_normals  != 0);
    ff.use_materials          = (m_params->use_materials != 0);
    ff.use_uv_seams           = (m_params->use_uv_seams  != 0);

    std::vector<int> mat_ids;
    if (ff.use_materials && m_input->material_ids) {
        mat_ids.assign(m_input->material_ids,
                       m_input->material_ids + m_nf);
    }

    m_features = detect_features(
        *m_mesh,
        ff.hard_edge_angle_deg,
        ff.use_normals,
        ff.use_materials,
        ff.use_uv_seams,
        mat_ids.empty() ? nullptr : &mat_ids,
        nullptr,                      // legacy uv_coords (unused)
        (ff.use_normals && m_input->normals) ? m_input->normals : nullptr,
        (ff.use_uv_seams && m_input->uv_indices) ? m_input->uv_indices : nullptr,
        3  // uv_face_stride: 3 indices per triangulated face
    );

    report_progress(0, 0.7f, "Preprocessing");

    // ------------------------------------------------------------------
    // Symmetry Detection — Roadmap §3.1 Step 1.4
    //
    // INTEGRATION (v101): detect_all_symmetry_axes() was implemented in
    // mesh/symmetry.cpp but was never called from the engine pipeline.
    // The three SymmetryResult values were never populated, so the symmetry
    // flags (symmetry_x/y/z) from QFParams had zero effect during Stage 1
    // — field/constraints.cpp received only boolean enforce_symmetry_* flags
    // but no vertex-correspondence or quality-score data.
    //
    // The BVH built here uses the *repaired* mesh so the correspondence
    // search works on clean geometry.  Only axes the user enabled are
    // tested to avoid unnecessary O(V log F) BVH queries.
    // ------------------------------------------------------------------
    {
        bool need_sym = (m_params->symmetry_x != 0) ||
                        (m_params->symmetry_y != 0) ||
                        (m_params->symmetry_z != 0);
        if (need_sym) {
            // Build Eigen matrices for TriangleBVH construction
            Eigen::MatrixXd Vs(m_mesh->num_vertices(), 3);
            for (int i = 0; i < m_mesh->num_vertices(); ++i) {
                Vec3 p = m_mesh->vertex_pos(i);
                Vs.row(i) << p.x(), p.y(), p.z();
            }
            Eigen::MatrixXi Fs(m_mesh->num_faces(), 3);
            for (int fi = 0; fi < m_mesh->num_faces(); ++fi) {
                auto fv = m_mesh->face_vertices(fi);
                Fs.row(fi) << fv[0], fv[1], fv[2];
            }
            TriangleBVH sym_bvh(Vs, Fs);

            SymmetryDetectionOptions sym_opts;
            sym_opts.tolerance            = 0.0;   // auto = 0.5% of bbox diagonal
            sym_opts.detection_threshold  = 0.95;  // roadmap: ">95% vertex correspondence"
            sym_opts.build_correspondence = true;
            // Cap sampling at 50K vertices so Stage 1 stays fast on dense meshes
            sym_opts.max_sample_vertices  =
                (m_mesh->num_vertices() > 50000) ? 50000 : 0;

            m_symmetry = detect_all_symmetry_axes(
                *m_mesh, sym_bvh,
                m_params->symmetry_x != 0,
                m_params->symmetry_y != 0,
                m_params->symmetry_z != 0,
                sym_opts);
        }
        // If no symmetry axis is requested, m_symmetry members keep their
        // default state (detected=false, score=0) — safe to read anywhere.
    }

    // Curvature
    m_curvature = compute_curvature(*m_mesh,
                                     resolve_thread_count(m_params->num_threads));

    // Sizing field
    Eigen::MatrixXf vc;
    const Eigen::MatrixXf* vc_ptr = nullptr;
    if (m_params->use_vertex_colors && m_input->vertex_colors) {
        vc.resize(m_nv, 3);
        for (int i = 0; i < m_nv; ++i) {
            vc(i,0) = m_input->vertex_colors[3*i+0];
            vc(i,1) = m_input->vertex_colors[3*i+1];
            vc(i,2) = m_input->vertex_colors[3*i+2];
        }
        vc_ptr = &vc;
    }

    m_sizing = compute_sizing_field(
        *m_mesh, m_curvature,
        m_params->target_quad_count,
        m_params->curvature_adaptivity,
        vc_ptr,
        m_features.hard_edges.empty() ? nullptr : &m_features.hard_edges);

    m_base_edge_len = (float)compute_base_edge_length(
        m_mesh->total_area(), m_params->target_quad_count);

    report_progress(0, 1.0f, "Preprocessing");
    return true;
}

// -----------------------------------------------------------------------
// Stage 2: Cross-field
// -----------------------------------------------------------------------

bool QFEngine::stage2_cross_field() {
    report_progress(1, 0.0f, "Computing cross-field");

    CrossFieldParams cfp;
    // BUG FIX (v83): The previous dispatch was:
    //   (field_solver == 2) ? CURVATURE_ONLY : KNOPPEL_2013
    // This silently mapped field_solver==0 (EIGEN_SMOOTH) to KNOPPEL_2013,
    // completely ignoring the user's algorithm choice.  Fixed to a three-way
    // dispatch matching the api.h comment: 0=EigenSmooth, 1=Knoppel2013, 2=Curvature-only.
    if (m_params->field_solver == 0)
        cfp.solver = FieldSolver::EIGEN_SMOOTH;
    else if (m_params->field_solver == 2)
        cfp.solver = FieldSolver::CURVATURE_ONLY;
    else
        cfp.solver = FieldSolver::KNOPPEL_2013;  // field_solver==1, or any unrecognised value
    cfp.num_threads    = m_params->num_threads;
    cfp.align_to_features = true;

    // FIX (v54+): Wire symmetry flags from QFParams into the field solver.
    // Previously these were declared in CrossFieldParams but never set from
    // QFParams, so X/Y/Z symmetry remeshing was silently ignored.
    cfp.enforce_symmetry_x = (m_params->symmetry_x != 0);
    cfp.enforce_symmetry_y = (m_params->symmetry_y != 0);
    cfp.enforce_symmetry_z = (m_params->symmetry_z != 0);

    m_field = compute_cross_field(*m_mesh, m_curvature, m_features, cfp);

    report_progress(1, 0.6f, "Computing cross-field");

    // FIX (v54+): Post-solve field smoothing pass.
    // The raw Knöppel eigenvector output can have residual noise near
    // feature edges (penalty terms cause local discontinuities) and near
    // degenerate triangles.  A short Gauss-Seidel smoothing pass on the
    // complex field representation suppresses this noise while respecting
    // feature constraints.
    //
    // Only run when the full Knöppel solver was used:
    //   - CURVATURE_ONLY produces smooth output by construction.
    //   - EIGEN_SMOOTH (v83) is a speed-optimised path — the Gauss-Seidel
    //     pass is intentionally skipped so it remains fast.
    if (cfp.solver == FieldSolver::KNOPPEL_2013 && m_mesh->num_faces() > 0) {
        // Rebuild the FieldConstraint list from detected feature edges
        // so the smoother knows which faces to protect.
        std::vector<FieldConstraint> field_constraints;
        if (!m_features.hard_edges.empty()) {
            // Reuse the edge-to-faces map already available from features
            // by converting the EdgeSet to the constraints.h flat format.
            std::vector<int32_t> feat_flat;
            feat_flat.reserve(m_features.hard_edges.size() * 2);
            for (auto& [lo, hi] : m_features.hard_edges) {
                feat_flat.push_back(lo);
                feat_flat.push_back(hi);
            }

            int nv = m_mesh->num_vertices();
            int nf = m_mesh->num_faces();

            std::vector<float> pos_f(nv * 3);
            std::vector<int32_t> tris_f(nf * 3);
            std::vector<float> e1_f(nf * 3), e2_f(nf * 3), fn_f(nf * 3);

            for (int vi = 0; vi < nv; ++vi) {
                auto p = m_mesh->vertex_pos(vi);
                pos_f[vi*3+0] = (float)p[0];
                pos_f[vi*3+1] = (float)p[1];
                pos_f[vi*3+2] = (float)p[2];
            }
            for (int fi = 0; fi < nf; ++fi) {
                auto [v0,v1,v2] = m_mesh->face_vertices(fi);
                tris_f[fi*3+0]=v0; tris_f[fi*3+1]=v1; tris_f[fi*3+2]=v2;
                // Local frames for constraint building.
                // BUG FIX (v83): The previous code called .normalized() directly on
                // face_normal() and on the v0→v1 edge without checking for zero length.
                // On degenerate (zero-area) triangles, face_normal() returns the zero
                // vector, and .normalized() gives NaN (0/0 in Eigen).  NaN then flows
                // into the constraint target and corrupts the Gauss-Seidel accumulator
                // for every face adjacent to the degenerate triangle.
                // Fix: mirror the guards used by compute_face_frames() in cross_field.cpp.
                Vec3 raw_n = m_mesh->face_normal(fi);
                double raw_n_len = raw_n.norm();
                Vec3 n = (raw_n_len > 1e-14) ? (raw_n / raw_n_len) : Vec3(0, 0, 1);

                // Project v0→v1 onto the tangent plane; fall back to unitOrthogonal()
                // when the edge is zero-length or parallel to n.
                Vec3 raw_edge = m_mesh->vertex_pos(v1) - m_mesh->vertex_pos(v0);
                Vec3 proj_edge = raw_edge - raw_edge.dot(n) * n;
                double proj_len = proj_edge.norm();
                Vec3 e1 = (proj_len > 1e-12) ? (proj_edge / proj_len) : n.unitOrthogonal();
                Vec3 e2 = n.cross(e1).normalized();

                fn_f[fi*3+0]=(float)n[0]; fn_f[fi*3+1]=(float)n[1]; fn_f[fi*3+2]=(float)n[2];
                e1_f[fi*3+0]=(float)e1[0]; e1_f[fi*3+1]=(float)e1[1]; e1_f[fi*3+2]=(float)e1[2];
                e2_f[fi*3+0]=(float)e2[0]; e2_f[fi*3+1]=(float)e2[1]; e2_f[fi*3+2]=(float)e2[2];
            }

            auto edge_to_faces = build_edge_to_faces(tris_f.data(), (int32_t)nf);
            build_feature_edge_constraints(
                pos_f.data(), tris_f.data(), (int32_t)nf,
                e1_f.data(), e2_f.data(), fn_f.data(),
                feat_flat, edge_to_faces,
                cfp.constraint_weight,
                field_constraints);
        }

        FieldSmoothingParams fsp;
        fsp.iterations       = 5;
        fsp.constraint_alpha = 0.95;
        fsp.convergence_tol  = 1e-6;
        fsp.num_threads      = m_params->num_threads;
        fsp.use_cotangent_weights = true;

        smooth_cross_field(*m_mesh, m_field, field_constraints, fsp);
    }

    report_progress(1, 0.85f, "Computing cross-field");

    // Detect singularities
    m_singularities = detect_singularities(*m_mesh, m_field);

    // INTEGRATION FIX (v82): Cache mesh + field + singularities for
    // qf_debug_field().  We deep-copy m_field and m_singularities (small) and
    // take a shallow clone of m_mesh via a second unique_ptr to the same object.
    // The debug cache is thread-local so concurrent remesh calls are isolated.
    // NOTE: We cannot move m_mesh here because stages 3-5 still need it.
    // Instead we copy-construct a new HalfEdgeMesh from the existing one.
    // The copy is O(V+F) — acceptable since it only runs once per remesh.
    {
        auto mesh_copy = std::make_unique<qf::HalfEdgeMesh>(*m_mesh);
        qf::CrossField field_copy = m_field;
        std::vector<qf::SingularityInfo> sings_copy = m_singularities;
        qf_cache_debug_state(std::move(mesh_copy),
                              std::move(field_copy),
                              std::move(sings_copy));
    }

    report_progress(1, 1.0f, "Computing cross-field");
    return true;
}

// -----------------------------------------------------------------------
// Stage 3: Parametrization
// -----------------------------------------------------------------------

bool QFEngine::stage3_parametrize() {
    report_progress(2, 0.0f, "Parametrizing surface");

    CombingResult combing = comb_field(*m_mesh, m_field, m_features);
    report_progress(2, 0.4f, "Parametrizing surface");

    // FIX BUG A + BUG B (v72):
    // Previously this called compute_minimal_seam_cut() from combing.h, which:
    //   (a) was a duplicate Dijkstra implementation of the proper seams.cpp code.
    //   (b) never forwarded uv_seam_edges from m_features — so use_uv_seams=1
    //       silently had zero effect on the parametrization seam cut.
    //
    // Replaced with compute_seam_cut_from_mesh() (seams.h) which:
    //   - Forwards combing inconsistency seams
    //   - Connects all singularities via Dijkstra
    //   - Forwards UV seam edges (BUG B FIX — previously dropped)
    //   - Forwards hard-edge seams as parametrization-friendly cut candidates
    EdgeSet seam_cut = compute_seam_cut_from_mesh(
        *m_mesh, combing, m_singularities, m_features);
    combing.seam_edges.insert(combing.seam_edges.end(),
                               seam_cut.begin(), seam_cut.end());

    ParamParams pp;
    // Map api param_method (0=MIQ, 1=IGM, 2=PoissonSimple) to internal enum
    if (m_params->param_method == 0)
        pp.method = ParamMethod::MIQ;
    else if (m_params->param_method == 1)
        pp.method = ParamMethod::IGM;
    else
        pp.method = ParamMethod::POISSON_SIMPLE;
    pp.num_threads = m_params->num_threads;

    m_uv = compute_parametrization(*m_mesh, combing, m_sizing, pp);

    report_progress(2, 1.0f, "Parametrizing surface");
    return true;
}

// -----------------------------------------------------------------------
// Stage 4: Quad extraction
// -----------------------------------------------------------------------

bool QFEngine::stage4_extract() {
    report_progress(3, 0.0f, "Extracting quads");

    // ----------------------------------------------------------------
    // Dispatch on extraction_method:
    //   0 = IsoLine  (default, fastest, high quality)
    //   1 = MotorcycleGraph  (iso-line base + aggressive T-junction fix)
    //   2 = DualContour      (QEF-optimal vertex placement, best near features)
    // ----------------------------------------------------------------
    const int method = m_params ? m_params->extraction_method : 0;

    const int n_threads = resolve_thread_count(m_params ? m_params->num_threads : 0);

    if (method == 1) {
        // --- Motorcycle Graph extraction (T-junction-free by construction) ---
        // Decomposes the surface into rectangular patches via the motorcycle
        // graph, parametrizes each patch independently (Tutte mapping), and
        // runs per-patch isoline extraction.  T-junctions cannot arise inside
        // a patch; only at patch boundaries, where they are resolved by the
        // vertex weld pass in extract_quads_motorcycle().
        m_quad_mesh = extract_quads_motorcycle(
            *m_mesh,
            m_field,
            m_uv,
            m_singularities,
            m_params ? m_params->target_quad_count : 5000,
            n_threads);

    } else if (method == 2) {
        // --- Dual Contouring (QEF-optimal vertex placement) ---
        // Best for hard-surface models with sharp feature edges.
        m_quad_mesh = extract_quads_dual_contour(
            *m_mesh, m_uv, n_threads);

    } else {
        // --- Seam-aware IsoLine extraction (method 0, default) ---
        // Uses the global UV parametrization with period-jump stitching
        // so that integer iso-lines are continuous across seam-cut edges.
        // Falls back to raw isoline tracing if uv.period_jumps is empty.
        m_quad_mesh = extract_quads_from_isolines_seam(
            *m_mesh, m_uv, n_threads);
    }

    report_progress(3, 0.4f, "Extracting quads");

    if (m_quad_mesh.quads.empty()) {
        // Fallback: copy input geometry as-is (no quad extraction)
        m_quad_mesh.vertices.resize(m_nv);
        for (int i = 0; i < m_nv; ++i)
            m_quad_mesh.vertices[i] = {m_pos[3*i], m_pos[3*i+1], m_pos[3*i+2]};
        // Convert tris to quad mesh tris
        int nf = (int)m_mesh->num_faces();
        m_quad_mesh.tris.resize(nf);
        for (int fi = 0; fi < nf; ++fi)
            m_quad_mesh.tris[fi] = m_mesh->face_vertices(fi);
    }

    // T-junction resolution (all methods — motorcycle graph pass)
    m_quad_mesh = resolve_t_junctions(*m_mesh, m_quad_mesh, m_singularities);

    report_progress(3, 0.65f, "Extracting quads");

    // Topology cleanup (degenerate removal, short edge collapse,
    // doublet removal, edge flips, hole filling)
    CleanupParams cp;
    cleanup_quad_mesh(m_quad_mesh, cp);

    report_progress(3, 0.85f, "Extracting quads");

    // ----------------------------------------------------------------
    // Edge Flow Refinement Pass (§13.10)
    //   Runs after cleanup — loop straightening, irregularity
    //   cancellation, and angle-based vertex relocation.
    //   Skipped for the Fast preset (method==2 already uses QEF which
    //   self-optimises vertex placement).
    // ----------------------------------------------------------------
    const bool fast_preset = m_params && (m_params->preset == 5);
    if (!fast_preset) {
        EdgeFlowRefineParams efp;
        efp.max_passes          = (method == 2) ? 1 : 2;
        efp.min_angle_deg       = 25.0;
        efp.zigzag_cos_thresh   = -0.3;
        efp.relocation_strength = 0.3;
        efp.preserve_features   = true;
        refine_edge_flow(m_quad_mesh, efp);
    }

    report_progress(3, 1.0f, "Extracting quads");
    return true;
}

// -----------------------------------------------------------------------
// Stage 5: Post-processing
// -----------------------------------------------------------------------

bool QFEngine::stage5_postprocess() {
    report_progress(4, 0.0f, "Post-processing");

    if (m_quad_mesh.vertices.empty()) {
        set_error("Stage 5: No output geometry");
        return false;
    }

    // Build BVH over input mesh
    Eigen::MatrixXd V(m_nv, 3);
    for (int i = 0; i < m_nv; ++i)
        V.row(i) << m_pos[3*i], m_pos[3*i+1], m_pos[3*i+2];

    Eigen::MatrixXi F(m_nt, 3);
    for (int fi = 0; fi < m_nt; ++fi) {
        auto fv = m_mesh->face_vertices(fi);
        F.row(fi) << fv[0], fv[1], fv[2];
    }
    TriangleBVH bvh(V, F);

    // Taubin smoothing interleaved with projection.
    // BUG FIX (v83 / Bug 4): The previous code computed sp.iterations from
    // m_params->smooth_iterations but then called iterative_smooth_and_project()
    // WITHOUT passing sp.iterations — the value was silently discarded.  Inside
    // iterative_smooth_and_project(), the hardcoded formula
    //   lround(10.0 / rounds)
    // was used regardless of the user's "Smooth Iterations" slider.  The slider
    // had zero effect on the number of smoothing passes applied.
    //
    // Fix: pass sp.iterations as the new smooth_iters_per_round parameter so the
    // user's setting is actually honoured.  The formula sp.iterations =
    // max(1, smooth_iterations/3) divides by 3 because iterative_smooth_and_project
    // runs 3 smooth+project cycles; dividing by 3 gives the correct per-cycle count.
    SmoothParams sp;
    sp.iterations  = std::max(1, m_params->smooth_iterations / 3);
    sp.lambda      = m_params->smooth_strength;
    sp.mu          = -m_params->smooth_strength * 1.06f;
    sp.num_threads = m_params->num_threads;

    iterative_smooth_and_project(
        m_quad_mesh, bvh, 3,
        sp.lambda, sp.mu,
        m_params->num_threads,
        sp.iterations);  // BUG FIX (v83): was missing — smooth_iterations slider now works

    report_progress(4, 0.5f, "Post-processing");

    // Feature snapping
    if (m_params->feature_snap_distance > 0.f)
        snap_to_features(m_quad_mesh, *m_mesh, m_features,
                         m_params->feature_snap_distance);

    // Final projection
    project_to_surface(m_quad_mesh, bvh, m_params->num_threads);

    // Tri→quad merge
    // BUG FIX (v95): convert_tris_to_quads() (subdiv.h) takes
    // const std::vector<double>&, but the previous code built pos_f as a
    // std::vector<float> and passed it directly — a type mismatch that fails
    // to compile.  QuadMesh::vertices stores std::array<double,3>, so we
    // build pos_d (double) directly from m_quad_mesh.vertices instead of
    // going through a float intermediate.  The float intermediate (pos_f) was
    // only used here; all later uses of pos_f in stage6 build their own
    // separate vector from m_quad_mesh.vertices at that point.
    std::vector<double> pos_d;
    pos_d.reserve(m_quad_mesh.vertices.size() * 3);
    for (auto& v : m_quad_mesh.vertices) {
        pos_d.push_back(v[0]);
        pos_d.push_back(v[1]);
        pos_d.push_back(v[2]);
    }
    // BUG FIX (v84 / B1): Renamed local vectors from `qf` and `tf` to
    // `quad_flat` and `tri_flat`.  Inside `namespace qf { ... }`, declaring a
    // local variable named `qf` shadows the enclosing namespace name — any
    // qualified use of `qf::SomeType` after that declaration would attempt to
    // resolve `SomeType` as a member of the std::vector, not the namespace.
    // While the current stage5 code happens to have no such qualified lookups
    // after the declaration, the shadowing is a latent compile-break hazard
    // for any future refactoring that adds one.
    std::vector<int32_t> quad_flat, tri_flat;
    for (auto& q : m_quad_mesh.quads)
        for (int v : q) quad_flat.push_back(v);
    for (auto& t : m_quad_mesh.tris)
        for (int v : t) tri_flat.push_back(v);
    convert_tris_to_quads(quad_flat, tri_flat, pos_d);
    // Put back
    m_quad_mesh.quads.clear();
    for (size_t i = 0; i + 3 < quad_flat.size(); i += 4)
        m_quad_mesh.quads.push_back({quad_flat[i], quad_flat[i+1], quad_flat[i+2], quad_flat[i+3]});
    m_quad_mesh.tris.clear();
    for (size_t i = 0; i + 2 < tri_flat.size(); i += 3)
        m_quad_mesh.tris.push_back({(int)tri_flat[i], (int)tri_flat[i+1], (int)tri_flat[i+2]});

    report_progress(4, 0.9f, "Post-processing");

    // Step 5.4 — Quality Metrics (was missing; quad_percentage and avg_valence
    // were always 0 in all outputs before this fix).
    //
    // BUG FIX (v83 / Bug 2): The previous call used the BASE overload:
    //   compute_quality_metrics(quad_mesh, features, target_quads,
    //                           snap_distance, num_threads)
    // which explicitly documents "feature alignment stays at −1 (not computed)
    // in this overload — callers that have ref_mesh should use the extended
    // overload."  Because m_mesh was available here all along, the extended
    // overload should have been called from day one.  The result was that
    // mean_feature_alignment_error_deg and max_feature_alignment_error_deg
    // were always −1.f in every QFResult, regardless of whether the mesh had
    // hard feature edges or not.
    //
    // Fix: call the extended overload with m_mesh.get() as ref_mesh.
    {
        QualityMetrics metrics = compute_quality_metrics(
            m_quad_mesh,
            &m_features,
            m_mesh.get(),                                         // BUG FIX (v83)
            m_params->target_quad_count,
            static_cast<double>(m_params->feature_snap_distance),
            m_params->num_threads);
        apply_metrics_to_mesh(m_quad_mesh, metrics);
    }

    report_progress(4, 1.0f, "Post-processing");
    return true;
}

// -----------------------------------------------------------------------
// Stage 6: Output packing
// -----------------------------------------------------------------------

bool QFEngine::stage6_output(QFResult* result) {
    report_progress(5, 0.0f, "Building output");

    int32_t nqv = (int32_t)m_quad_mesh.vertices.size();
    if (nqv == 0) { set_error("Stage 6: No output vertices"); return false; }

    // INTEGRATION FIX (v82): Build flat C arrays and delegate to pack_result()
    // from mesh_io.cpp.  Benefits:
    //   1. Quad normals now use the diagonal cross-product method (better than
    //      splitting each quad into two triangles and summing — which caused
    //      visible normal seams at the v0/v2 diagonal on non-planar quads).
    //   2. Leftover tri normals are also accumulated (previously only quads
    //      contributed, giving incorrect normals at quad/tri boundaries).
    //   3. material_ids is zero-initialised inside pack_result() so the field
    //      is always valid on return (was previously unset here).
    //   4. A single consistent implementation for all packing across the engine.

    // --- Build flat input arrays ---
    std::vector<float> pos_f;
    pos_f.reserve(nqv * 3);
    for (auto& v : m_quad_mesh.vertices) {
        pos_f.push_back((float)v[0]);
        pos_f.push_back((float)v[1]);
        pos_f.push_back((float)v[2]);
    }

    std::vector<int32_t> quad_faces_f;
    quad_faces_f.reserve(m_quad_mesh.quads.size() * 4);
    for (auto& q : m_quad_mesh.quads)
        for (int v : q) quad_faces_f.push_back(v);

    std::vector<int32_t> tri_leftover_f;
    tri_leftover_f.reserve(m_quad_mesh.tris.size() * 3);
    for (auto& t : m_quad_mesh.tris)
        for (int v : t) tri_leftover_f.push_back(v);

    // --- Delegate to mesh_io::pack_result() ---
    // Fills: positions, faces, tri_faces, normals, material_ids,
    //        num_vertices, num_quad_faces, num_tri_faces,
    //        quad_percentage, avg_valence (geometry-derived estimate).
    pack_result(pos_f, quad_faces_f, tri_leftover_f, result);

    // BUG FIX (v84 / B2): quad_percentage unit mismatch.
    //
    // pack_result() sets result->quad_percentage in [0, 100] (percentage, e.g.
    // 97.4 for 97.4% quads).  QualityMetrics::quad_percentage is in [0, 1]
    // (fraction).  apply_metrics_to_mesh() copies the fraction directly into
    // m_quad_mesh.quad_percentage, so the override below must multiply by 100
    // to restore the [0, 100] scale that bridge.py and the UI panels expect.
    //
    // Before this fix, a mesh with 97.4% quads would display as "0.97%" in the
    // Blender panel — 100× smaller than the correct value.
    result->quad_percentage = m_quad_mesh.quad_percentage * 100.f;
    result->avg_valence     = m_quad_mesh.avg_valence;

    // BUG FIX (v84 / B3): Material ID transfer — finally implemented.
    //
    // pack_result() allocates result->material_ids as a zero-initialised array
    // of size num_quad_faces and its comment said "engine.cpp fills these after
    // the call."  Before this fix, engine.cpp never did, so result->material_ids
    // was always all-zeros regardless of what per-face materials the input mesh
    // had.  Material-based feature detection (use_materials=1) worked — those
    // boundary edges were correctly detected in Stage 1 — but the materials were
    // never propagated to the output, meaning all remeshed faces came back with
    // material_id=0 and the artist's material assignments were silently discarded.
    //
    // Algorithm:
    //   1. Build the reverse map: tri_to_poly[ti] = original input face fi,
    //      from m_poly_to_tri (populated in stage1 by triangulate_input()).
    //   2. Rebuild the BVH over the triangulated input mesh (same construction
    //      as in stage5_postprocess — cheap because the mesh is already loaded).
    //   3. For each output quad, compute its centroid, query the BVH for the
    //      nearest input triangle, and copy that triangle's material_id.
    //
    // Cost: O(nqf * log(nt)) — negligible compared to the solve time.
    // Gated on m_input->material_ids != nullptr to avoid wasted BVH construction
    // for the common case where no materials are assigned.
    if (m_input->material_ids != nullptr && result->num_quad_faces > 0
        && !m_poly_to_tri.empty() && m_mesh)
    {
        // Build tri → original-face reverse map
        std::vector<int32_t> tri_to_poly(m_nt, 0);
        for (int fi = 0; fi < m_nf; ++fi) {
            int sz = m_face_sizes ? m_face_sizes[fi] : 3;
            // Number of triangles produced by triangulate_input for this face:
            //   tri  (sz==3) → 1   quad (sz==4) → 2   ngon (sz>4) → sz-2
            int n_tris = (sz == 3) ? 1 : (sz == 4) ? 2 : (sz > 4) ? (sz - 2) : 0;
            int start  = m_poly_to_tri[fi];
            for (int k = 0; k < n_tris && (start + k) < m_nt; ++k)
                tri_to_poly[static_cast<size_t>(start + k)] = fi;
        }

        // Build BVH over input triangulated mesh
        Eigen::MatrixXd V(m_nv, 3);
        for (int i = 0; i < m_nv; ++i)
            V.row(i) << m_pos[3*i], m_pos[3*i+1], m_pos[3*i+2];
        Eigen::MatrixXi F(m_nt, 3);
        for (int fi = 0; fi < m_nt; ++fi) {
            auto fv = m_mesh->face_vertices(fi);
            F.row(fi) << fv[0], fv[1], fv[2];
        }
        TriangleBVH mat_bvh(V, F);

        // Transfer material IDs
        int32_t nqf = result->num_quad_faces;
        for (int32_t qi = 0; qi < nqf; ++qi) {
            // Compute quad centroid from result positions
            Vec3 centroid{0.0, 0.0, 0.0};
            for (int k = 0; k < 4; ++k) {
                int vi = result->faces[qi * 4 + k];
                if (vi >= 0 && vi < result->num_vertices) {
                    centroid[0] += result->positions[vi * 3 + 0];
                    centroid[1] += result->positions[vi * 3 + 1];
                    centroid[2] += result->positions[vi * 3 + 2];
                }
            }
            centroid *= 0.25;

            auto hit = mat_bvh.closest_point(centroid);
            if (hit.face_index >= 0 && hit.face_index < m_nt) {
                int orig_fi = tri_to_poly[static_cast<size_t>(hit.face_index)];
                result->material_ids[qi] = m_input->material_ids[orig_fi];
            }
        }
    }

    report_progress(5, 1.0f, "Done");
    return true;
}

} // namespace qf
