"""QuadForge engine package — complete geometry processing pipeline.

Modules:
  Core mesh:     halfedge, topology, spatial
  Analysis:      curvature, features, constraints, sizing
  Field:         field, field_smoothing, sparse_solver
  Combing/Param: combing, parametrize
  Extraction:    extraction, quad_merger, feature_protection, motorcycle
  Singularity:   singularity
  Symmetry:      symmetry
  Post-process:  smoothing, projection, snapping, transfer, output_enhance
  Metrics:       metrics
  Subdivision:   subdiv
  Pipeline:      pipeline, exact_count
  v1.1 Incremental:      selection_remesh
  v1.3 Neural Singularity: neural_singularity
  v1.4 Live Preview:     live_preview
  v2.0 Multi-Resolution: multiresolution

Dominance Layer (Roadmap Part XI + Part XIII):
  Repair:        mesh_repair           (§11.2, §13.7)
  Determinism:   determinism           (§13.1)
  Artist AI:     artist_intelligence   (§11.4, §13.2)
  Optimizer:     quality_optimizer     (§13.3)
  Confidence:    confidence            (§11.6, §13.9)
  Edge flow:     edge_flow_refine      (§13.10)
  Preview:       progressive_preview   (§13.8)
  Classifier:    mesh_classifier       (§11.3, §13.5)
  Integrator:    dominance_pipeline
"""

from .pipeline import run_pipeline, run_pipeline_exact_count
from .halfedge import HalfEdgeMesh
from .topology import analyze_topology, TopologyInfo
from .spatial import KDTree, TriangleBVH
from .sizing import compute_sizing_field, compute_base_edge_length
from .curvature import compute_curvature, CurvatureData
from .field import compute_cross_field, CrossFieldData
from .combing import comb_field, compute_seam_cut
from .parametrize import parametrize_poisson, apply_miq_rounding
from .extraction import trace_isolines, cleanup_quad_mesh
from .projection import project_to_surface, iterative_smooth_and_project
from .snapping import snap_to_features
from .transfer import transfer_materials
from .metrics import compute_quality_metrics, QualityMetrics
from .symmetry import detect_symmetry, SymmetryData, enforce_output_symmetry
from .singularity import optimise_singularities
from .motorcycle import resolve_t_junctions, detect_t_junctions
from .output_enhance import compute_output_normals
from .subdiv import check_subdiv_compatibility, SubdivReport
from .selection_remesh import (
    extract_selection,
    run_selection_pipeline,
    stitch_selection,
    SelectionData,
    StitchResult,
    build_selection_mask_from_face_indices,
)
from .live_preview import compute_field_preview, FieldPreview
# v1.3 — Neural Singularity Placement
from .neural_singularity import (
    predict_singularity_locations,
    build_vertex_features,
    nms_singularities,
    neural_optimise_singularities,
)
# v2.0 — Multi-Resolution Pipeline
from .multiresolution import (
    run_multiresolution_pipeline,
    run_adaptive_pipeline,
    decimate_mesh,
    prolongate_field,
    prolongate_uv,
    MULTIRESOLUTION_THRESHOLD,
)

# =====================================================================
# v4.0 — Dominance Layer (Roadmap Parts XI + XIII)
# =====================================================================
from .mesh_repair import (
    repair_mesh,
    voxel_rescue_remesh,
    RepairReport,
)
from .determinism import (
    DeterministicContext,
    canonicalize_mesh,
    mesh_fingerprint,
    params_fingerprint,
    is_deterministic_mode,
    get_rng,
    deterministic_sum,
    stable_argsort,
    deterministic_choose,
)
from .artist_intelligence import (
    classify_mesh,
    detect_semantic_features,
    lookup_template,
    apply_template_bias,
    run_artist_intelligence,
    MeshClass,
    SemanticFeatures,
    TopologyTemplate,
)
from .mesh_classifier import (
    SolverTier,
    DispatchDecision,
    TierRunResult,
    decide_tier,
    run_tier,
    dispatch_with_fallback,
)
from .quality_optimizer import (
    QualityBreakdown,
    OptimizerCandidate,
    OptimizerResult,
    compute_quality_score,
    run_auto_optimizer,
    QUALITY_WEIGHTS,
    DEFAULT_SWEEP,
)
from .confidence import (
    QualityGates,
    ConfidenceReport,
    RetryPlan,
    check_quality_gates,
    compute_face_risk,
    compute_confidence,
    plan_retry,
)
from .edge_flow_refine import (
    EdgeFlowReport,
    detect_irregular_clusters,
    remove_doublets,
    collapse_short_edge_loops,
    relax_non_features,
    refine_edge_flow,
)
from .progressive_preview import (
    PreviewCache,
    PreviewTier,
    PreviewOverlays,
    build_tier1_preview,
    build_tier2_preview,
    run_progressive_preview,
    compute_preview_fingerprint,
)
from .dominance_pipeline import (
    run_dominance_pipeline,
    run_dominance_pipeline_compat,
    DominanceResult,
)


__all__ = [
    # Core pipeline
    "run_pipeline", "run_pipeline_exact_count",
    # Mesh
    "HalfEdgeMesh", "analyze_topology", "TopologyInfo",
    "KDTree", "TriangleBVH",
    # Analysis
    "compute_sizing_field", "compute_base_edge_length",
    "compute_curvature", "CurvatureData",
    "compute_cross_field", "CrossFieldData",
    # Param + extraction
    "comb_field", "compute_seam_cut",
    "parametrize_poisson", "apply_miq_rounding",
    "trace_isolines", "cleanup_quad_mesh",
    # Post-process
    "project_to_surface", "iterative_smooth_and_project",
    "snap_to_features", "transfer_materials",
    # Analysis tools
    "compute_quality_metrics", "QualityMetrics",
    "detect_symmetry", "SymmetryData", "enforce_output_symmetry",
    "optimise_singularities",
    "resolve_t_junctions", "detect_t_junctions",
    "compute_output_normals",
    "check_subdiv_compatibility", "SubdivReport",
    # v1.1 Incremental remeshing
    "extract_selection",
    "run_selection_pipeline",
    "stitch_selection",
    "SelectionData",
    "StitchResult",
    "build_selection_mask_from_face_indices",
    # v1.3 Neural Singularity Placement
    "predict_singularity_locations",
    "build_vertex_features",
    "nms_singularities",
    "neural_optimise_singularities",
    # v1.4 Live preview
    "compute_field_preview",
    "FieldPreview",
    # v2.0 Multi-Resolution Pipeline
    "run_multiresolution_pipeline",
    "run_adaptive_pipeline",
    "decimate_mesh",
    "prolongate_field",
    "prolongate_uv",
    "MULTIRESOLUTION_THRESHOLD",
    # Dominance Layer — repair
    "repair_mesh", "voxel_rescue_remesh", "RepairReport",
    # Dominance Layer — determinism
    "DeterministicContext", "canonicalize_mesh", "mesh_fingerprint",
    "params_fingerprint", "is_deterministic_mode", "get_rng",
    "deterministic_sum", "stable_argsort", "deterministic_choose",
    # Dominance Layer — artist intelligence
    "classify_mesh", "detect_semantic_features", "lookup_template",
    "apply_template_bias", "run_artist_intelligence",
    "MeshClass", "SemanticFeatures", "TopologyTemplate",
    # Dominance Layer — quality optimizer
    "compute_quality_score", "run_auto_optimizer",
    "QualityBreakdown", "OptimizerCandidate", "OptimizerResult",
    "QUALITY_WEIGHTS", "DEFAULT_SWEEP",
    # Dominance Layer — confidence & gates
    "QualityGates", "ConfidenceReport", "check_quality_gates",
    "compute_face_risk", "compute_confidence", "plan_retry", "RetryPlan",
    # Dominance Layer — edge flow refinement
    "refine_edge_flow", "detect_irregular_clusters", "remove_doublets",
    "collapse_short_edge_loops", "relax_non_features", "EdgeFlowReport",
    # Dominance Layer — progressive preview
    "run_progressive_preview", "build_tier1_preview", "build_tier2_preview",
    "compute_preview_fingerprint",
    "PreviewCache", "PreviewTier", "PreviewOverlays",
    # Dominance Layer — mesh classifier / dispatcher
    "decide_tier", "dispatch_with_fallback", "run_tier",
    "DispatchDecision", "TierRunResult", "SolverTier",
    # Dominance Layer — integrator
    "run_dominance_pipeline", "run_dominance_pipeline_compat",
    "DominanceResult",
]
