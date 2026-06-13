"""QuadForge main property definitions.

QFSettingsPropertyGroup holds every user-facing parameter:
  - Target & Sizing
  - Feature Detection
  - Symmetry
  - Smoothing
  - Performance
  - Algorithm Selection (field solver, parametrization, extraction)
  - Read-only result properties (populated after each remesh)
"""

import bpy
from bpy.props import (
    BoolProperty,
    FloatProperty,
    IntProperty,
    EnumProperty,
    PointerProperty,
    StringProperty,
)


def _on_preset_changed(self, context):
    """Called when the preset dropdown changes — applies the preset values."""
    from .presets import apply_preset
    apply_preset(self, self.preset)


class QFSettingsPropertyGroup(bpy.types.PropertyGroup):
    """QuadForge main settings — stored on bpy.types.Scene."""

    preset: EnumProperty(
        name="Preset",
        description="Quick configuration preset — overrides settings below",
        items=[
            ('CUSTOM',       "Custom",       "Manually tuned settings"),
            ('ORGANIC',      "Organic",      "Smooth surfaces: characters, creatures, natural forms"),
            ('HARD_SURFACE', "Hard Surface", "Sharp edges: weapons, vehicles, mechanical parts"),
            ('SCULPT',       "Sculpt",       "Sculpted high-poly meshes needing retopology"),
            ('ARCHITECTURE', "Architecture", "Buildings, rooms — strict right angles"),
            ('FAST',         "Fast",         "Speed over quality — useful for quick preview"),
        ],
        default='CUSTOM',
        update=_on_preset_changed,
    )

    # --- Target & Sizing ---
    target_quad_count: IntProperty(
        name="Target Quad Count",
        description="Desired number of quads in the remeshed result",
        default=5000, min=100, max=1_000_000,
    )
    curvature_adaptivity: FloatProperty(
        name="Curvature Adaptivity",
        description="Higher values allocate more quads to high-curvature areas (0=uniform, 1=max)",
        default=0.5, min=0.0, max=1.0, subtype='FACTOR',
    )
    exact_quad_count: BoolProperty(
        name="Exact Quad Count",
        description="Binary-search to hit the exact target quad count (3-5x slower)",
        default=False,
    )
    use_vertex_colors: BoolProperty(
        name="Use Vertex Color Density",
        description="Use active vertex color as density map (red=finer, green=coarser)",
        default=False,
    )
    density_paint_strength: FloatProperty(
        name="Paint Strength",
        description=(
            "Strength of the vertex color density effect.\n"
            "1.0 = full 4× scaling at maximum red/green; 0.0 = no effect"
        ),
        default=1.0, min=0.0, max=1.0, subtype='FACTOR',
    )

    # --- Feature Detection ---
    auto_detect_hard_edges: BoolProperty(
        name="Auto Detect Hard Edges",
        description="Detect sharp edges by dihedral angle and align quads to them",
        default=True,
    )
    # BUG FIX: removed subtype='ANGLE' — it caused Blender to store radians
    # internally while the engine expects degrees. Plain float stores degrees.
    hard_edge_angle_deg: FloatProperty(
        name="Hard Edge Angle (°)",
        description="Dihedral angle threshold for hard edge detection in degrees",
        default=30.0, min=1.0, max=180.0,
    )
    use_normals: BoolProperty(
        name="Use Normal Splits",
        description="Treat custom split-normal discontinuities as feature edges",
        default=False,
    )
    use_materials: BoolProperty(
        name="Use Materials",
        description="Treat material boundaries as feature edges",
        default=False,
    )
    use_uv_seams: BoolProperty(
        name="Use UV Seams",
        description="Treat UV seam edges as features (QuadForge-exclusive)",
        default=False,
    )
    feature_snap_distance: FloatProperty(
        name="Feature Snap Distance",
        description="Max distance to snap output vertices onto feature curves",
        default=0.1, min=0.0, max=10.0, subtype='DISTANCE',
    )

    # --- Symmetry ---
    symmetry_x: BoolProperty(name="X", description="Enforce symmetry across local X axis", default=False)
    symmetry_y: BoolProperty(name="Y", description="Enforce symmetry across local Y axis", default=False)
    symmetry_z: BoolProperty(name="Z", description="Enforce symmetry across local Z axis", default=False)

    # --- Smoothing ---
    smooth_iterations: IntProperty(
        name="Smooth Iterations",
        description="Number of Taubin bilaplacian smoothing passes on the output",
        default=10, min=0, max=100,
    )
    smooth_strength: FloatProperty(
        name="Smooth Strength",
        description="Taubin smoothing lambda (how strongly each pass moves vertices)",
        default=0.5, min=0.0, max=1.0, subtype='FACTOR',
    )

    # --- Performance ---
    num_threads: IntProperty(
        name="Threads",
        description="CPU threads for parallel stages (0 = auto-detect)",
        default=0, min=0, max=64,
    )

    # --- Algorithm Selection (Advanced) ---
    field_solver: EnumProperty(
        name="Field Solver",
        description=(
            "Algorithm for the 4-RoSy cross-field solve.\n"
            "Knoppel 2013: globally optimal eigensolver (best quality, recommended).\n"
            "Eigen Smooth: fast Laplacian-smoothed field — good balance.\n"
            "Curvature Only: fastest, no global solve, lower quality."
        ),
        items=[
            ('KNOPPEL',      "Knöppel 2013",   "Globally optimal eigensolver — best quality (recommended)"),
            ('EIGEN_SMOOTH', "Eigen Smooth",    "Laplacian-smoothed curvature field — fast & good quality"),
            ('CURVATURE',    "Curvature Only",  "Purely curvature-aligned field — fastest, lower quality"),
        ],
        default='KNOPPEL',
    )
    param_method: EnumProperty(
        name="Parametrization",
        description=(
            "Global surface parametrization method.\n"
            "MIQ: cleanest edge loops via mixed-integer rounding (recommended).\n"
            "IGM: Integer-Grid Maps — strictest rectangular layout, best for CAD/architecture.\n"
            "Poisson Simple: fastest, no integer rounding, looser alignment."
        ),
        items=[
            ('MIQ',     "MIQ",           "Mixed-Integer Quadrangulation — cleanest loops (recommended)"),
            ('IGM',     "IGM",           "Integer-Grid Maps — strictest rectangular grid, ideal for CAD"),
            ('POISSON', "Poisson Simple", "Fast Poisson-only parametrization — no MIQ rounding"),
        ],
        default='MIQ',
    )
    extraction_method: EnumProperty(
        name="Extraction",
        description=(
            "Quad extraction strategy.\n"
            "Iso-line: trace integer UV iso-lines — best topology (recommended).\n"
            "Motorcycle Graph: motorcycle-graph T-junction elimination — cleanest corners.\n"
            "Dual Contour: dual contouring from parametrization — experimental.\n"
            "Greedy Merge: fast triangle-pair merging — lowest quality fallback."
        ),
        items=[
            ('ISO',        "Iso-line",        "Trace integer iso-curves — best overall topology (recommended)"),
            ('MOTORCYCLE', "Motorcycle Graph", "Motorcycle-graph extraction — T-junction-free, cleanest corners"),
            ('DUAL',       "Dual Contour",     "Dual contouring from UV grid — experimental"),
            ('GREEDY',     "Greedy Merge",     "Direct triangle-pair merging — fastest, lowest quality"),
        ],
        default='ISO',
    )

    # --- GPU Acceleration (v1.2) ---
    use_gpu: BoolProperty(
        name="GPU Acceleration",
        description=(
            "Use GPU-accelerated sparse conjugate gradient solver for the "
            "cross-field and parametrization stages.\n"
            "Requires NVIDIA CUDA, Apple Metal, or OpenCL.\n"
            "Falls back to CPU if no compatible GPU is detected."
        ),
        default=False,
        update=lambda self, ctx: None,
    )
    gpu_backend: EnumProperty(
        name="GPU Backend",
        description=(
            "GPU solver backend to use when GPU acceleration is enabled.\n"
            "Auto: detect and select the best available backend.\n"
            "CUDA: NVIDIA GPUs (fastest on supported hardware).\n"
            "Metal: Apple Silicon/AMD on macOS.\n"
            "OpenCL: cross-platform GPU (requires pyopencl).\n"
            "CPU: force CPU-only (disables GPU even if available)."
        ),
        items=[
            ('AUTO',   "Auto",   "Detect and select best available backend"),
            ('CUDA',   "CUDA",   "NVIDIA CUDA — fastest on NVIDIA hardware"),
            ('METAL',  "Metal",  "Apple Metal — macOS only"),
            ('OPENCL', "OpenCL", "OpenCL — cross-platform, requires pyopencl"),
            ('CPU',    "CPU",    "Force CPU-only solver (no GPU)"),
        ],
        default='AUTO',
    )


    # --- Neural Singularity Placement (v1.3) ---
    use_neural_singularity: BoolProperty(
        name="Neural Singularity Placement",
        description=(
            "Use the v1.3 learned singularity scorer to predict optimal irregular-vertex "
            "positions based on local geometry, rather than relying solely on the "
            "topological heuristic.\n"
            "Blends neural predictions with topology-driven placement via the blend slider."
        ),
        default=False,
    )
    neural_singularity_blend: FloatProperty(
        name="Neural Blend",
        description=(
            "Blend factor between topological (0.0) and neural (1.0) singularity placement.\n"
            "0.0 = pure topological heuristic (original behaviour).\n"
            "0.5 = equal mix — recommended starting point.\n"
            "1.0 = pure neural prediction."
        ),
        default=0.5, min=0.0, max=1.0, subtype='FACTOR',
    )

    # --- Multi-Resolution Pipeline (v2.0) ---
    use_multiresolution: BoolProperty(
        name="Multi-Resolution Mode",
        description=(
            "Use the v2.0 hierarchical coarse-to-fine pipeline for large meshes.\n"
            "Automatically activated for meshes above ~200K triangles; can be forced "
            "on for any mesh to reduce computation time at a slight quality cost."
        ),
        default=False,
    )

    # --- Output Options ---
    shade_smooth_output: BoolProperty(
        name="Shade Smooth Output",
        description="Apply smooth shading to the remeshed result",
        default=True,
    )
    keep_original: BoolProperty(
        name="Keep Original",
        description="Keep the input mesh visible (do not auto-hide after remeshing)",
        default=True,
    )

    # =====================================================================
    # Dominance Layer (Roadmap Part XI + Part XIII)
    # =====================================================================
    enable_dominance_mode: BoolProperty(
        name="Dominance Mode",
        description=(
            "Enable the full Dominance Layer pipeline: mesh repair, "
            "classification, adaptive dispatch, quality gates with retry, "
            "optional auto optimizer, and edge-flow refinement"
        ),
        default=True,
    )
    enable_mesh_repair: BoolProperty(
        name="Mesh Repair",
        description=(
            "Run the mesh repair & sanitization layer before remeshing. "
            "Welds duplicates, removes degenerates, fixes winding, splits "
            "non-manifold vertices"
        ),
        default=True,
    )
    enable_artist_intelligence: BoolProperty(
        name="Artist Intelligence",
        description=(
            "Classify the mesh and apply topology templates to bias "
            "singularity placement toward artist-friendly edge flow"
        ),
        default=True,
    )
    enable_auto_optimizer: BoolProperty(
        name="Auto Quality Optimizer",
        description=(
            "After the main solve, search nearby parameter variations "
            "and keep the best result by composite quality score. "
            "Adds up to ~40%% of the base solve time"
        ),
        default=False,
    )
    enable_edge_flow_refine: BoolProperty(
        name="Edge Flow Refinement",
        description=(
            "Final local refinement pass that removes doublets, collapses "
            "sliver quads, and applies feature-preserving relaxation"
        ),
        default=True,
    )
    reproducible_mode: BoolProperty(
        name="Reproducible Mode",
        description=(
            "Deterministic quality mode. Pins all random seeds and forces "
            "single-threaded solves so the same input produces bit-exactly "
            "the same output every time"
        ),
        default=False,
    )
    reproducible_seed: IntProperty(
        name="Seed",
        description="Seed used when Reproducible Mode is enabled",
        default=0xC0FFEE,
        min=0,
        max=0x7FFFFFFF,
    )
    confidence_target: FloatProperty(
        name="Confidence Target",
        description=(
            "Minimum acceptable confidence score (0–100). Results below "
            "this threshold trigger an automatic retry with safer settings"
        ),
        default=75.0,
        min=0.0,
        max=100.0,
        subtype='PERCENTAGE',
    )
    time_budget_seconds: FloatProperty(
        name="Time Budget (s)",
        description=(
            "Maximum wall-clock time to spend on a single remesh. The "
            "dispatcher downgrades to simpler solvers if the primary "
            "pipeline would exceed this budget"
        ),
        default=60.0,
        min=1.0,
        max=3600.0,
        subtype='TIME',
    )

    # --- Read-only results (written by _build_output) ---
    last_quad_count:      IntProperty(name="Quads",        default=0)
    last_tri_count:       IntProperty(name="Tris",         default=0)
    last_quad_percentage: FloatProperty(name="Quad %",     default=0.0)
    last_avg_valence:     FloatProperty(name="Avg Valence", default=0.0)
    last_elapsed:         FloatProperty(name="Time (s)",   default=0.0)
    last_min_jacobian:    FloatProperty(name="Min SJ",     default=0.0)
    last_mean_jacobian:   FloatProperty(name="Mean SJ",    default=0.0)
    last_input_object:    StringProperty(name="Source",    default="")

    # --- Dominance Layer read-only results ---
    last_confidence_score: FloatProperty(
        name="Confidence",
        description="Final confidence score of the last remesh (0–100)",
        default=0.0, min=0.0, max=100.0, subtype='PERCENTAGE',
    )
    last_mesh_category: StringProperty(name="Category", default="")
    last_solver_tier:   StringProperty(name="Solver",   default="")
    last_retries_used:  IntProperty(name="Retries",     default=0)
    last_repair_welded: IntProperty(name="Welded",      default=0)
    last_repair_fixed:  IntProperty(name="Repaired",    default=0)
    last_gate_status:   StringProperty(name="Gates",    default="")


classes = (QFSettingsPropertyGroup,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    if not hasattr(bpy.types.Scene, "quadforge_settings"):
        bpy.types.Scene.quadforge_settings = PointerProperty(type=QFSettingsPropertyGroup)


def unregister():
    if hasattr(bpy.types.Scene, "quadforge_settings"):
        del bpy.types.Scene.quadforge_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
