"""QuadForge UI panels — complete sidebar layout.

Panel hierarchy (all parented to QUADFORGE_PT_main):
  QUADFORGE_PT_main          — Header + Preset + Remesh button / progress bar
  QUADFORGE_PT_target        — Target & Sizing
  QUADFORGE_PT_features      — Feature Detection
  QUADFORGE_PT_symmetry      — Symmetry
  QUADFORGE_PT_smoothing     — Smoothing / Post-processing
  QUADFORGE_PT_performance   — Performance
  QUADFORGE_PT_advanced      — Advanced Algorithm Selection (collapsed)
  QUADFORGE_PT_results       — Results (shown only after a remesh)
  QUADFORGE_PT_info          — Info / About / Reset
"""

import bpy

# Import _state so the panel can show a live progress bar while running
from .operators import _state as _remesh_state
# Import custom icon helper (returns 0 gracefully if icons not loaded)
from . import icon_id as _icon_id

VERSION = "17.0.0"


# ======================================================================
# Main panel — header, preset, big remesh button / progress bar
# ======================================================================

class QUADFORGE_PT_main(bpy.types.Panel):
    bl_label = "QuadForge"
    bl_idname = "QUADFORGE_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "QuadForge"

    def draw(self, context):
        layout = self.layout

        if not hasattr(context.scene, "quadforge_settings"):
            layout.label(text="Settings not initialized", icon='ERROR')
            return

        settings = context.scene.quadforge_settings
        obj = context.active_object
        has_mesh = obj is not None and obj.type == 'MESH'

        # --- Preset selector row with icon buttons ---
        row = layout.row(align=True)
        row.prop(settings, "preset", text="")

        # Icon shortcut buttons for each preset (gracefully skip if icons absent)
        preset_icons = {
            'ORGANIC':      'organic',
            'HARD_SURFACE': 'hard_surface',
            'SCULPT':       'sculpt',
            'ARCHITECTURE': 'architecture',
            'FAST':         'fast',
        }
        sub = layout.row(align=True)
        sub.scale_x = 0.85
        for preset_key, icon_name in preset_icons.items():
            ico = _icon_id(icon_name)
            op = sub.operator(
                "quadforge.apply_preset",
                text="",
                icon_value=ico if ico else 0,
                icon='MESH_GRID' if not ico else 'NONE',
            )
            op.preset = preset_key
            #op.depress = (settings.preset == preset_key)

        layout.separator(factor=0.5)

        # --- Selection Remesh button (edit mode only) ---
        if context.mode == 'EDIT_MESH':
            box = layout.box()
            col = box.column(align=True)
            col.label(text="Selection Remesh (v1.1)", icon='FACE_MAPS')
            col.operator("quadforge.remesh_selection",
                         text="⚡ Remesh Selected Faces",
                         icon='FACE_MAPS')
            layout.separator(factor=0.5)

        # --- Main action: either live progress bar or Remesh button ---
        box = layout.box()
        if _remesh_state.running:
            # Running — show progress bar replacing the button
            col = box.column(align=True)
            col.label(
                text=f"⏳ {_remesh_state.stage_name}  "
                     f"({_remesh_state.progress * 100:.0f}%)",
                icon='TIME',
            )
            # Unicode block-character progress bar
            row = col.row(align=True)
            filled = max(1, int(_remesh_state.progress * 10))
            bar = "█" * filled + "░" * (10 - filled)
            row.label(text=bar)
            col.label(text="Press ESC to cancel", icon='EVENT_ESC')
        else:
            col = box.column(align=True)
            col.scale_y = 1.8
            if has_mesh:
                label = f"⚡ REMESH IT  ({settings.target_quad_count:,} quads)"
            else:
                label = "⚡ REMESH IT"
            col.operator("quadforge.remesh", text=label, icon='MESH_GRID')

            if not has_mesh:
                box.label(text="Select a mesh object", icon='INFO')


# ======================================================================
# Target & Sizing
# ======================================================================

class QUADFORGE_PT_target(bpy.types.Panel):
    bl_label = "Target & Sizing"
    bl_idname = "QUADFORGE_PT_target"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "QuadForge"
    bl_parent_id = "QUADFORGE_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.quadforge_settings

        layout.prop(settings, "target_quad_count")
        layout.prop(settings, "curvature_adaptivity", slider=True)
        layout.prop(settings, "exact_quad_count")

        layout.separator(factor=0.5)
        row = layout.row(align=True)
        row.prop(settings, "use_vertex_colors")
        if settings.use_vertex_colors:
            row.operator("quadforge.paint_density", text="", icon='BRUSH_DATA')
            col = layout.column(align=True)
            col.prop(settings, "density_paint_strength", slider=True)


# ======================================================================
# Feature Detection
# ======================================================================

class QUADFORGE_PT_features(bpy.types.Panel):
    bl_label = "Feature Detection"
    bl_idname = "QUADFORGE_PT_features"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "QuadForge"
    bl_parent_id = "QUADFORGE_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.quadforge_settings

        col = layout.column(align=True)
        col.prop(settings, "auto_detect_hard_edges")
        sub = col.column()
        sub.enabled = settings.auto_detect_hard_edges
        sub.prop(settings, "hard_edge_angle_deg")

        layout.separator(factor=0.5)
        layout.prop(settings, "use_normals")
        layout.prop(settings, "use_materials")
        layout.prop(settings, "use_uv_seams")

        layout.separator(factor=0.5)
        layout.prop(settings, "feature_snap_distance")


# ======================================================================
# Symmetry
# ======================================================================

class QUADFORGE_PT_symmetry(bpy.types.Panel):
    bl_label = "Symmetry"
    bl_idname = "QUADFORGE_PT_symmetry"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "QuadForge"
    bl_parent_id = "QUADFORGE_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.quadforge_settings

        row = layout.row(align=True)
        row.prop(settings, "symmetry_x", toggle=True)
        row.prop(settings, "symmetry_y", toggle=True)
        row.prop(settings, "symmetry_z", toggle=True)
        layout.label(text="Uses object local coordinates", icon='INFO')


# ======================================================================
# Smoothing
# ======================================================================

class QUADFORGE_PT_smoothing(bpy.types.Panel):
    bl_label = "Smoothing"
    bl_idname = "QUADFORGE_PT_smoothing"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "QuadForge"
    bl_parent_id = "QUADFORGE_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.quadforge_settings

        layout.prop(settings, "smooth_iterations")
        layout.prop(settings, "smooth_strength", slider=True)


# ======================================================================
# Performance
# ======================================================================

class QUADFORGE_PT_performance(bpy.types.Panel):
    bl_label = "Performance"
    bl_idname = "QUADFORGE_PT_performance"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "QuadForge"
    bl_parent_id = "QUADFORGE_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.quadforge_settings

        layout.prop(settings, "num_threads")

        layout.separator(factor=0.4)

        # GPU Acceleration (v1.2) ----------------------------------------
        gpu_box = layout.box()
        row = gpu_box.row()
        row.label(text="GPU Acceleration (v1.2)", icon='OUTLINER_OB_LIGHT')

        row2 = gpu_box.row()
        row2.prop(settings, "use_gpu", toggle=True,
                  icon='FUND' if settings.use_gpu else 'X',
                  text="Enable GPU Solver" if not settings.use_gpu else "GPU Solver: ON")

        if settings.use_gpu:
            gpu_box.prop(settings, "gpu_backend")
            # Live GPU status row
            sub = gpu_box.row()
            sub.enabled = False
            try:
                from .engine.gpu_solver import gpu_status_string
                sub.label(text=gpu_status_string(), icon='INFO')
            except Exception:
                sub.label(text="GPU status unavailable", icon='INFO')

            detect_row = gpu_box.row()
            detect_row.operator("quadforge.detect_gpu",
                                text="Detect GPU", icon='VIEWZOOM')

        # ---------------------------------------------------------------

        layout.separator(factor=0.5)
        layout.label(text="Output", icon='EXPORT')
        layout.prop(settings, "shade_smooth_output")
        layout.prop(settings, "keep_original")



# ======================================================================
# Advanced — Algorithm Selection
# ======================================================================

class QUADFORGE_PT_advanced(bpy.types.Panel):
    bl_label = "Advanced"
    bl_idname = "QUADFORGE_PT_advanced"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "QuadForge"
    bl_parent_id = "QUADFORGE_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.quadforge_settings

        layout.label(text="Algorithm Selection", icon='SETTINGS')

        box = layout.box()
        col = box.column(align=True)
        col.prop(settings, "field_solver")
        col.prop(settings, "param_method")
        col.prop(settings, "extraction_method")

        layout.separator(factor=0.5)
        layout.label(
            text="Knöppel + MIQ + Iso: best quality",
            icon='INFO',
        )
        layout.label(
            text="Curvature + Poisson + Greedy: fastest",
            icon='INFO',
        )

        # ----- v1.3 Neural Singularity Placement -----
        layout.separator(factor=0.5)
        neural_box = layout.box()
        neural_row = neural_box.row(align=True)
        neural_row.label(text="Neural Singularity (v1.3)", icon='OUTLINER_OB_POINTCLOUD')
        neural_row.operator(
            "wm.url_open",
            text="",
            icon='QUESTION',
            emboss=False,
        ).url = "https://github.com/reynold/quadforge/blob/main/docs/ALGORITHMS.md#v13-neural-singularity"
        neural_box.prop(settings, "use_neural_singularity", toggle=True,
                        icon='SHADERFX' if settings.use_neural_singularity else 'SHADING_WIRE',
                        text=("Neural Placement: ON" if settings.use_neural_singularity
                              else "Neural Placement: OFF"))
        if settings.use_neural_singularity:
            neural_box.prop(settings, "neural_singularity_blend", slider=True)
            neural_box.label(
                text="0 = topology only  |  1 = neural only",
                icon='INFO',
            )

        layout.separator(factor=0.5)
        layout.label(text="Debug", icon='TOOL_SETTINGS')
        layout.operator(
            "quadforge.visualize_field",
            text="Visualize Cross-Field",
            icon='NORMALS_VERTEX_FACE',
        )
        layout.operator(
            "quadforge.check_environment",
            text="Check Environment",
            icon='SCRIPT',
        )

        layout.separator(factor=0.3)
        layout.label(text="Subdivision Readiness", icon='MOD_SUBSURF')
        row = layout.row(align=True)
        row.operator(
            "quadforge.check_subdiv",
            text="Check CC Compatibility",
            icon='CHECKMARK',
        )
        sub = row.row(align=True)
        sub.scale_x = 0.6
        # Inline strict toggle — accesses the operator's property via a search
        # workaround: show a separate button for Strict mode
        layout.operator(
            "quadforge.check_subdiv",
            text="Check (Strict)",
            icon='ERROR',
        ).strict = True




# ======================================================================
# Multi-Resolution panel (v2.0)
# ======================================================================

class QUADFORGE_PT_multiresolution(bpy.types.Panel):
    bl_label   = "Multi-Resolution (v2.0)"
    bl_idname  = "QUADFORGE_PT_multiresolution"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "QuadForge"
    bl_parent_id   = "QUADFORGE_PT_main"
    bl_options     = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        self.layout.label(text="", icon='MESH_DATA')

    def draw(self, context):
        layout = self.layout
        settings = context.scene.quadforge_settings

        col = layout.column(align=True)
        col.label(
            text="Hierarchical coarse-to-fine remeshing.",
            icon='INFO',
        )
        col.label(text="Best for meshes > 200K triangles (3–5× faster).")

        layout.separator(factor=0.4)
        layout.prop(settings, "use_multiresolution", toggle=True,
                    icon='MOD_DECIM' if settings.use_multiresolution else 'MOD_DECIM',
                    text=("Multi-Res Mode: ON" if settings.use_multiresolution
                          else "Multi-Res Mode: OFF"))

        layout.separator(factor=0.4)
        row = layout.row()
        row.scale_y = 1.4
        row.operator(
            "quadforge.remesh_multiresolution",
            text="⚡ Remesh (Multi-Res)",
            icon='MESH_DATA',
        )

        layout.separator(factor=0.3)
        obj = context.active_object
        if obj and obj.type == 'MESH':
            n_poly = len(obj.data.polygons)
            if n_poly < 200_000:
                box = layout.box()
                box.label(
                    text=f"Current mesh: {n_poly:,} faces",
                    icon='INFO',
                )
                box.label(text="Multi-Res is most useful above 200K faces.")
            else:
                box = layout.box()
                box.label(
                    text=f"Large mesh: {n_poly:,} faces ✓",
                    icon='CHECKMARK',
                )
                box.label(text="Multi-Res pipeline recommended.")

# ======================================================================
# Results  — shown only after at least one remesh
# ======================================================================

# ======================================================================
# Dominance Layer panel (Roadmap Part XI + Part XIII)
# ======================================================================

class QUADFORGE_PT_dominance(bpy.types.Panel):
    bl_label   = "Dominance Layer"
    bl_idname  = "QUADFORGE_PT_dominance"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "QuadForge"
    bl_parent_id   = "QUADFORGE_PT_main"
    bl_options     = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        self.layout.label(text="", icon='SHADERFX')

    def draw(self, context):
        layout = self.layout
        settings = context.scene.quadforge_settings

        col = layout.column(align=True)
        col.label(
            text="Stability + intelligence + quality gates.",
            icon='INFO',
        )
        col.label(text="Recommended ON for production meshes.")

        layout.separator(factor=0.4)
        layout.prop(
            settings, "enable_dominance_mode",
            toggle=True,
            icon='CHECKMARK' if settings.enable_dominance_mode else 'X',
            text=("Dominance Mode: ON" if settings.enable_dominance_mode
                  else "Dominance Mode: OFF"),
        )

        dm = settings.enable_dominance_mode

        # Stage toggles
        box = layout.box()
        box.enabled = dm
        box.label(text="Pipeline Stages", icon='NODETREE')
        col = box.column(align=True)
        col.prop(settings, "enable_mesh_repair",         icon='MODIFIER')
        col.prop(settings, "enable_artist_intelligence", icon='OUTLINER_OB_ARMATURE')
        col.prop(settings, "enable_auto_optimizer",      icon='SORT_DESC')
        col.prop(settings, "enable_edge_flow_refine",    icon='MOD_SMOOTH')

        # Quality gates + confidence
        box = layout.box()
        box.enabled = dm
        box.label(text="Quality Gates", icon='CHECKBOX_HLT')
        col = box.column(align=True)
        col.prop(settings, "confidence_target",  slider=True)
        col.prop(settings, "time_budget_seconds")

        # Determinism
        box = layout.box()
        box.enabled = dm
        box.label(text="Determinism", icon='LOCKED')
        col = box.column(align=True)
        col.prop(settings, "reproducible_mode",
                 icon='LOCKED' if settings.reproducible_mode else 'UNLOCKED')
        sub = col.row(align=True)
        sub.enabled = dm and settings.reproducible_mode
        sub.prop(settings, "reproducible_seed")

        # Last-run results
        if settings.last_confidence_score > 0 or settings.last_mesh_category:
            box = layout.box()
            box.label(text="Last Run", icon='INFO')
            grid = box.grid_flow(row_major=True, columns=2,
                                 even_columns=True, align=True)
            grid.label(text=f"Category: {settings.last_mesh_category or '—'}")
            grid.label(text=f"Solver: {settings.last_solver_tier or '—'}")
            grid.label(text=f"Confidence: {settings.last_confidence_score:.0f}")
            grid.label(text=f"Retries: {settings.last_retries_used}")
            grid.label(text=f"Welded: {settings.last_repair_welded}")
            grid.label(text=f"Repaired: {settings.last_repair_fixed}")
            if settings.last_gate_status:
                box.label(text=f"Gates: {settings.last_gate_status}")


# ======================================================================
# Live Preview panel (v1.4)
# ======================================================================

class QUADFORGE_PT_live_preview(bpy.types.Panel):
    bl_label   = "Live Field Preview (v1.4)"
    bl_idname  = "QUADFORGE_PT_live_preview"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "QuadForge"
    bl_parent_id   = "QUADFORGE_PT_advanced"
    bl_options     = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.label(
            text="Preview cross-field without full remesh",
            icon='HIDE_OFF',
        )
        row = layout.row(align=True)
        op = row.operator("quadforge.live_preview",
                          text="Show Field Preview", icon='HIDE_OFF')
        layout.label(
            text="Creates '_QF_Preview_Field' object.",
            icon='INFO',
        )
        layout.label(
            text="Delete that object to remove the overlay.",
            icon='INFO',
        )


class QUADFORGE_PT_results(bpy.types.Panel):
    bl_label = "Results"
    bl_idname = "QUADFORGE_PT_results"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "QuadForge"
    bl_parent_id = "QUADFORGE_PT_main"

    @classmethod
    def poll(cls, context):
        if not hasattr(context.scene, "quadforge_settings"):
            return False
        s = context.scene.quadforge_settings
        return s.last_quad_count > 0 or s.last_tri_count > 0

    def draw(self, context):
        layout = self.layout
        settings = context.scene.quadforge_settings

        col = layout.column(align=True)
        col.label(
            text=f"Quads:  {settings.last_quad_count:,}  "
                 f"({settings.last_quad_percentage:.1f}%)",
            icon='MESH_GRID',
        )
        col.label(text=f"Triangles:  {settings.last_tri_count:,}")
        col.label(text=f"Avg Valence:  {settings.last_avg_valence:.3f}")

        layout.separator(factor=0.5)
        col2 = layout.column(align=True)
        col2.label(
            text=f"Scaled Jacobian:  "
                 f"{settings.last_min_jacobian:.3f} min  /  "
                 f"{settings.last_mean_jacobian:.3f} mean"
        )
        col2.label(
            text=f"Time:  {settings.last_elapsed:.2f}s",
            icon='TIME',
        )

        if settings.last_input_object:
            layout.label(
                text=f"Source:  {settings.last_input_object}",
                icon='OBJECT_DATA',
            )


# ======================================================================
# Info / About
# ======================================================================

class QUADFORGE_PT_info(bpy.types.Panel):
    bl_label = "Info"
    bl_idname = "QUADFORGE_PT_info"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "QuadForge"
    bl_parent_id = "QUADFORGE_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.label(text=f"QuadForge  v{VERSION}", icon='INFO')
        layout.label(text="By Reynold — GPL v3")
        layout.separator(factor=0.5)
        row = layout.row(align=True)
        row.operator(
            "wm.url_open",
            text="Documentation",
            icon='HELP',
        ).url = "https://github.com/reynold/quadforge/blob/main/docs/USER_GUIDE.md"
        row.operator(
            "wm.url_open",
            text="GitHub",
            icon='URL',
        ).url = "https://github.com/reynold/quadforge"
        layout.separator(factor=0.5)
        layout.operator("quadforge.reset_defaults", text="Reset to Defaults", icon='FILE_REFRESH')
        layout.separator(factor=0.3)
        layout.operator("quadforge.check_environment", text="Check Environment", icon='SCRIPT')


# ======================================================================
# Registration
# ======================================================================

classes = (
    QUADFORGE_PT_main,
    QUADFORGE_PT_target,
    QUADFORGE_PT_features,
    QUADFORGE_PT_symmetry,
    QUADFORGE_PT_smoothing,
    QUADFORGE_PT_performance,
    QUADFORGE_PT_advanced,
    QUADFORGE_PT_live_preview,        # v1.4
    QUADFORGE_PT_multiresolution,      # v2.0
    QUADFORGE_PT_dominance,            # Dominance Layer (XI+XIII)
    QUADFORGE_PT_results,
    QUADFORGE_PT_info,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
