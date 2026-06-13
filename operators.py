"""QuadForge operators.

QUADFORGE_OT_remesh       — Modal operator: background thread, progress bar,
                             ESC cancellation, shade-smooth transfer, keep/hide input.
QUADFORGE_OT_reset_defaults — Reset all settings to their default values.
QUADFORGE_OT_paint_density  — Activate vertex-color paint mode for density map.
"""

import bpy
import numpy as np
import bmesh
import time
import threading
import traceback
from .bridge import (
    remesh as _bridge_remesh,
    extract_mesh,
    build_params,
    build_output_mesh,
)
from .engine.metrics import compute_quality_metrics


# ==========================================================
# Shared state for the background thread
# ==========================================================

class _RemeshState:
    """Thread-safe state shared between the modal timer and the worker."""
    def __init__(self):
        self.running    = False
        self.abort      = False
        self.finished   = False
        self.error: str | None = None
        self.stage_name = ""
        self.progress   = 0.0
        self.result_verts         = None
        self.result_faces         = None
        self.result_uvs           = None   # (V_out, 2) float32 or None
        self.result_vertex_colors = None   # (V_out, 3) float32 or None
        self.result_normals       = None   # (V_out, 3) float32 or None
        self.elapsed    = 0.0
        self.settings_snapshot = None  # _SettingsSnapshot used by the worker (for Dominance report writeback)


_state = _RemeshState()


class _PipelineAborted(Exception):
    """Raised inside _progress_callback when the user presses ESC."""


def _progress_callback(stage: str, progress: float):
    """Called from the engine thread — writes to shared state only.

    BUG FIX v29: also checks _state.abort so that pressing ESC actually
    terminates the pipeline.  Previously _state.abort was set by modal()
    but never read here, so the pipeline ran to completion regardless.
    """
    _state.stage_name = stage
    _state.progress   = progress
    if _state.abort:
        raise _PipelineAborted("Cancelled by user")


def _worker_thread(input_mesh, params, obj_name, settings_snapshot):
    """Run the pipeline in a background thread."""
    global _state
    try:
        t0 = time.perf_counter()

        result = _bridge_remesh(
            input_mesh,
            params,
            progress_cb=_progress_callback,
        )

        # Support both the new _PipelineResult object and legacy (verts, faces) tuple
        if hasattr(result, 'verts'):
            _state.result_verts         = result.verts
            _state.result_faces         = result.faces
            _state.result_uvs           = getattr(result, 'output_uvs',           None)
            _state.result_vertex_colors = getattr(result, 'output_vertex_colors', None)
            _state.result_normals       = getattr(result, 'output_normals',       None)
        else:
            # run_pipeline_exact_count returns (verts, faces, actual_count)
            # run_pipeline (legacy) returns (verts, faces) — handle both
            if isinstance(result, tuple):
                verts = result[0]
                faces = result[1]
            else:
                verts, faces = result, result  # fallback (should not happen)
            _state.result_verts         = verts
            _state.result_faces         = faces
            _state.result_uvs           = None
            _state.result_vertex_colors = None
            _state.result_normals       = None

        _state.elapsed  = time.perf_counter() - t0
        _state.finished = True
    except _PipelineAborted:
        # BUG FIX v29: user pressed ESC — treat as a clean cancellation,
        # not an error.  Don't set _state.error so modal()/_finish() reports
        # "No output produced" rather than a red error message.
        print("[QuadForge] Pipeline cancelled by user (ESC)")
        _state.result_verts = None
        _state.result_faces = None
        _state.finished = True
    except Exception as e:
        traceback.print_exc()
        _state.error    = str(e)
        _state.finished = True


# ==========================================================
# Settings snapshot (plain Python, no bpy dependency)
# ==========================================================

class _SettingsSnapshot:
    """Plain-Python copy of Blender settings for thread safety."""
    def __init__(self, settings):
        self.auto_detect_hard_edges = bool(settings.auto_detect_hard_edges)
        self.hard_edge_angle_deg    = float(settings.hard_edge_angle_deg)
        self.use_normals            = bool(settings.use_normals)
        self.use_materials          = bool(settings.use_materials)
        self.use_uv_seams           = bool(settings.use_uv_seams)
        self.use_vertex_colors      = bool(settings.use_vertex_colors)
        self.target_quad_count      = int(settings.target_quad_count)
        self.curvature_adaptivity   = float(settings.curvature_adaptivity)
        self.exact_quad_count       = bool(settings.exact_quad_count)
        self.smooth_iterations      = int(settings.smooth_iterations)
        self.smooth_strength        = float(settings.smooth_strength)
        self.feature_snap_distance  = float(settings.feature_snap_distance)
        self.symmetry_x             = bool(settings.symmetry_x)
        self.symmetry_y             = bool(settings.symmetry_y)
        self.symmetry_z             = bool(settings.symmetry_z)
        # BUG FIX v29: pipeline.py reads symmetry via getattr(params, 'symmetry', ...)
        # as a single (x, y, z) tuple.  _SettingsSnapshot only stored the three
        # separate booleans, so getattr() always returned (False, False, False) on
        # the background-thread path, silently disabling symmetry for every large mesh.
        self.symmetry               = (self.symmetry_x, self.symmetry_y, self.symmetry_z)
        self.num_threads            = int(settings.num_threads)
        self.preset                 = str(settings.preset)
        # Algorithm selection
        self.field_solver      = str(getattr(settings, 'field_solver',      'KNOPPEL'))
        self.param_method      = str(getattr(settings, 'param_method',      'MIQ'))
        self.extraction_method = str(getattr(settings, 'extraction_method', 'ISO'))
        # Output options
        self.shade_smooth_output = bool(getattr(settings, 'shade_smooth_output', True))
        self.keep_original       = bool(getattr(settings, 'keep_original',       True))
        # v1.3 Neural Singularity
        self.use_neural_singularity  = bool(getattr(settings, 'use_neural_singularity',  False))
        self.neural_singularity_blend = float(getattr(settings, 'neural_singularity_blend', 0.5))
        # v2.0 Multi-Resolution
        self.use_multiresolution = bool(getattr(settings, 'use_multiresolution', False))
        # -----------------------------------------------------------------
        # v4.0 — Dominance Layer (Roadmap Part XI + XIII) — INPUT FLAGS.
        # BUG FIX v42: these were missing, so the threaded path always saw
        # enable_dominance_mode=False via getattr's fallback and the entire
        # Dominance Layer silently never ran for any mesh large enough to
        # trigger background-thread execution.
        # -----------------------------------------------------------------
        self.enable_dominance_mode      = bool(getattr(settings, 'enable_dominance_mode',      False))
        self.enable_mesh_repair         = bool(getattr(settings, 'enable_mesh_repair',         True))
        self.enable_artist_intelligence = bool(getattr(settings, 'enable_artist_intelligence', True))
        self.enable_auto_optimizer      = bool(getattr(settings, 'enable_auto_optimizer',      True))
        self.enable_edge_flow_refine    = bool(getattr(settings, 'enable_edge_flow_refine',    True))
        self.reproducible_mode          = bool(getattr(settings, 'reproducible_mode',          False))
        self.reproducible_seed          = int(getattr(settings,  'reproducible_seed',          42))
        self.confidence_target          = float(getattr(settings,'confidence_target',          70.0))
        self.time_budget_seconds        = float(getattr(settings,'time_budget_seconds',        30.0))
        # -----------------------------------------------------------------
        # Dominance Layer — REPORT FIELDS (written by
        # dominance_pipeline._write_result_to_settings onto THIS snapshot
        # during the worker thread). _build_output() copies them back to
        # the live PropertyGroup on the main thread after the worker ends.
        # -----------------------------------------------------------------
        self.last_confidence_score = 0.0
        self.last_mesh_category    = ""
        self.last_solver_tier      = ""
        self.last_retries_used     = 0
        self.last_repair_welded    = 0
        self.last_repair_fixed     = 0
        self.last_gate_status      = ""


# ==========================================================
# QUADFORGE_OT_remesh — Modal operator
# ==========================================================

class QUADFORGE_OT_remesh(bpy.types.Operator):
    """Run QuadForge quad remeshing on the active mesh object."""
    bl_idname  = "quadforge.remesh"
    bl_label   = "QuadForge Remesh"
    bl_description = "Run QuadForge quad remeshing on the active mesh"
    bl_options = {'REGISTER', 'UNDO'}

    _timer  = None
    _thread = None

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None
            and obj.type == 'MESH'
            and obj.data is not None
            and context.mode == 'OBJECT'
            and not _state.running
        )

    def execute(self, context):
        global _state
        obj      = context.active_object
        mesh     = obj.data
        settings = context.scene.quadforge_settings

        # --- Input validation ---
        if len(mesh.vertices) == 0 or len(mesh.polygons) == 0:
            self.report({'ERROR'}, "Active mesh has no valid geometry")
            return {'CANCELLED'}
        if len(mesh.vertices) < 4:
            self.report({'ERROR'}, "Mesh needs at least 4 vertices")
            return {'CANCELLED'}

        # --- Warn about non-manifold geometry ---
        bm = bmesh.new()
        bm.from_mesh(mesh)

        non_manifold = any(len(e.link_faces) != 2 for e in bm.edges)
        bm.free()
        if non_manifold:
            self.report({'WARNING'},
                "QuadForge: mesh has non-manifold edges — results may be imperfect")

        # --- Extract mesh (requires bpy main thread) ---
        try:
            input_mesh = extract_mesh(obj)
        except Exception as e:
            self.report({'ERROR'}, f"Mesh extraction failed: {e}")
            return {'CANCELLED'}

        # --- Extract features (requires bmesh — main thread only) ---
        from .engine.features import extract_features
        try:
            features = extract_features(obj, settings)
        except Exception:
            features = None

        params = build_params(settings)
        snap   = _SettingsSnapshot(settings)

        if features is not None:
            input_mesh._features = features

        # --- Log ---
        print(f"[QuadForge] {'='*44}")
        print(f"[QuadForge] Object:      {obj.name}")
        print(f"[QuadForge] Input:       {len(mesh.vertices)} verts, {len(mesh.polygons)} faces")
        print(f"[QuadForge] Target:      {settings.target_quad_count} quads")
        print(f"[QuadForge] Preset:      {settings.preset}")
        print(f"[QuadForge] Algorithms:  "
              f"field={params.field_solver}  param={params.param_method}  "
              f"extract={params.extraction_method}")

        # --- Reset state ---
        _state.__init__()
        _state.running = True

        # Small meshes → synchronous (simpler, no thread issues).
        # The sync path passes the LIVE settings PropertyGroup directly
        # into run_pipeline, so dominance_pipeline writes its report
        # fields straight to the PG and no snapshot writeback is needed.
        if len(mesh.polygons) < 2000:
            return self._run_sync(context, obj, input_mesh, params, settings, features)

        # BUG FIX v42: On the threaded path the worker sees the snapshot,
        # not the live PG. Stash it on _state so _build_output() can copy
        # the Dominance Layer report fields (confidence, retries, repair
        # counts, mesh category, solver tier, gate status) from the
        # snapshot back onto the live PropertyGroup on the main thread
        # after the worker ends.
        _state.settings_snapshot = snap

        # --- Launch background thread ---
        self._thread = threading.Thread(
            target=_worker_thread,
            args=(input_mesh, params, obj.name, snap),
            daemon=True,
        )
        self._thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)

        self.report({'INFO'}, "QuadForge: Remeshing started…")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        global _state

        if event.type == 'TIMER':
            # Redraw every VIEW_3D area so the N-panel progress bar updates
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()

            # Update the header bar on the operator's own area
            # Guard: context.area is None when called outside a viewport region.
            try:
                if context.area is not None:
                    context.area.header_text_set(
                        f"QuadForge: {_state.stage_name} "
                        f"({_state.progress * 100:.0f}%)  [ESC to cancel]"
                    )
            except Exception:
                pass

            if _state.finished:
                return self._finish(context)

        if event.type == 'ESC':
            _state.abort = True
            self.report({'WARNING'}, "QuadForge: Cancelling…")
            return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def _finish(self, context):
        global _state

        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if context.area is not None:
            try:
                context.area.header_text_set(None)
            except Exception:
                pass
        _state.running = False

        if _state.error:
            self.report({'ERROR'}, f"QuadForge failed: {_state.error}")
            return {'CANCELLED'}

        if (_state.result_verts is None or _state.result_faces is None
                or len(_state.result_verts) == 0
                or len(_state.result_faces) == 0):
            self.report({'ERROR'}, "QuadForge: No output produced")
            return {'CANCELLED'}

        return self._build_output(
            context,
            context.active_object,
            _state.result_verts,
            _state.result_faces,
            _state.elapsed,
            output_uvs=_state.result_uvs,
            output_vertex_colors=_state.result_vertex_colors,
            output_normals=_state.result_normals,
        )

    def _run_sync(self, context, obj, input_mesh, params, settings, features):
        global _state
        try:
            t0 = time.perf_counter()
            result = _bridge_remesh(
                   input_mesh,
                   params,
                  progress_cb=_progress_callback,

                )
            elapsed = time.perf_counter() - t0
            _state.running = False

            if hasattr(result, 'verts'):
                return self._build_output(
                    context, obj, result.verts, result.faces, elapsed,
                    output_uvs=getattr(result, 'output_uvs', None),
                    output_vertex_colors=getattr(result, 'output_vertex_colors', None),
                    output_normals=getattr(result, 'output_normals', None),
                )
            elif isinstance(result, tuple):
                # run_pipeline_exact_count → (verts, faces, actual_count)
                # run_pipeline legacy     → (verts, faces)
                verts = result[0]
                faces = result[1]
                return self._build_output(context, obj, verts, faces, elapsed)
            else:
                self.report({'ERROR'}, "QuadForge: Unexpected pipeline result type")
                return {'CANCELLED'}
        except _PipelineAborted:
            # BUG FIX v29: sync path also needs a clean abort path.
            print("[QuadForge] Pipeline cancelled by user (ESC)")
            _state.running = False
            return {'CANCELLED'}
        except Exception as e:
            traceback.print_exc()
            _state.running = False
            self.report({'ERROR'}, f"QuadForge failed: {e}")
            return {'CANCELLED'}

    def _build_output(
        self, context, obj,
        result_verts, result_faces, elapsed,
        output_uvs=None,
        output_vertex_colors=None,
        output_normals=None,
    ):
        """Create the Blender output mesh, apply shading, transfer materials, UVs, and vertex colors."""
        settings = context.scene.quadforge_settings

        # Build base mesh (vertices + faces only)
        new_mesh = build_output_mesh(
            name=f"{obj.name}_QuadForge",
            vertices=result_verts,
            faces=result_faces,
        )

        # ---- Apply transferred UV map ----
        if output_uvs is not None and len(output_uvs) > 0:
            try:
                uv_layer = new_mesh.uv_layers.new(name="QuadForge_UV")
                # BUG-H FIX (v117): the previous per-loop Python double-loop
                #   for face in result_faces: for vi in face: uv_per_loop[i] = uvs[vi]
                # was O(total_loops) pure-Python — ~40K iterations at ~1 µs each = 40 ms
                # for a 10K-quad mesh, scaling linearly to 400 ms at 100K quads.
                # Fix: build a flat vertex-index array with np.concatenate, then
                # use advanced numpy indexing to expand UVs to per-loop in one call.
                uv_arr    = np.asarray(output_uvs, dtype=np.float32)  # (V, 2)
                vi_flat   = np.concatenate([np.asarray(f, dtype=np.int32)
                                            for f in result_faces])   # (total_loops,)
                # Clamp indices to valid range (guard against OOB faces)
                vi_flat   = np.clip(vi_flat, 0, len(uv_arr) - 1)
                uv_per_loop = uv_arr[vi_flat]                         # (total_loops, 2)
                uv_layer.data.foreach_set("uv", uv_per_loop.ravel())
                print(f"[QuadForge] Applied UV map: {len(uv_per_loop)} loops")
            except Exception as e:
                print(f"[QuadForge] UV apply skipped: {e}")

        # ---- Apply transferred vertex colors ----
        if output_vertex_colors is not None and len(output_vertex_colors) > 0:
            try:
                num_verts = len(result_verts)
                # Blender 4.x uses color_attributes (POINT domain preferred)
                ca = new_mesh.color_attributes.new(
                    name="Col",
                    type='FLOAT_COLOR',
                    domain='POINT',
                )
                colors_rgba = np.ones((num_verts, 4), dtype=np.float32)
                colors_rgba[:, :3] = output_vertex_colors[:num_verts, :3]
                ca.data.foreach_set("color", colors_rgba.ravel())
                print(f"[QuadForge] Applied vertex colors: {num_verts} vertices")
            except Exception as e:
                print(f"[QuadForge] Vertex color apply skipped: {e}")

        # Create and link object
        new_obj = bpy.data.objects.new(f"{obj.name}_QuadForge", new_mesh)
        context.collection.objects.link(new_obj)
        new_obj.matrix_world = obj.matrix_world.copy()

        # Transfer materials
        for slot in obj.material_slots:
            if slot.material:
                new_obj.data.materials.append(slot.material)

        # ---- Shade smooth / flat based on settings (fast path via foreach_set) ----
        shade_smooth = getattr(settings, 'shade_smooth_output', True)
        num_out_polys = len(new_mesh.polygons)
        if shade_smooth:
            smooth_arr = np.ones(num_out_polys, dtype=bool)
        else:
            # Mirror the input: use smooth only if input mesh used it
            in_smooth = np.empty(len(obj.data.polygons), dtype=bool)
            obj.data.polygons.foreach_get("use_smooth", in_smooth)
            input_smooth = bool(in_smooth.any())
            smooth_arr = np.full(num_out_polys, input_smooth, dtype=bool)
        new_mesh.polygons.foreach_set("use_smooth", smooth_arr)
        new_mesh.update()

        # Select output, deselect everything else
        bpy.ops.object.select_all(action='DESELECT')
        new_obj.select_set(True)
        context.view_layer.objects.active = new_obj

        # ---- Keep or hide input ----
        keep_original = getattr(settings, 'keep_original', True)
        if not keep_original:
            obj.hide_set(True)

        # Also check addon preference for auto-hide (legacy support)
        if keep_original:
            try:
                prefs = context.preferences.addons.get("QuadForge")
                if prefs and prefs.preferences.auto_hide_input:
                    obj.hide_set(True)
            except Exception:
                pass

        # ---- Compute & store quality metrics ----
        metrics = compute_quality_metrics(result_verts, result_faces)
        settings.last_quad_count      = metrics.num_quads
        settings.last_tri_count       = metrics.num_tris
        settings.last_quad_percentage = metrics.quad_percentage
        settings.last_avg_valence     = metrics.avg_valence
        settings.last_elapsed         = elapsed
        settings.last_min_jacobian    = metrics.min_scaled_jacobian
        settings.last_mean_jacobian   = metrics.mean_scaled_jacobian
        if hasattr(settings, 'last_input_object'):
            settings.last_input_object = obj.name

        # ----------------------------------------------------------------
        # BUG FIX v42: Dominance Layer report writeback.
        # On the threaded path, dominance_pipeline wrote its report fields
        # (confidence, retries, repair counts, category, solver tier, gate
        # status) onto the _SettingsSnapshot — which is a plain Python
        # object, not the live PropertyGroup. Copy those values back so
        # the N-panel "Last Run" block actually reflects what happened.
        # On the sync path this is a harmless no-op because the pipeline
        # already wrote directly to the live settings.
        # ----------------------------------------------------------------
        snap = getattr(_state, 'settings_snapshot', None)
        if snap is not None:
            for fname in (
                'last_confidence_score',
                'last_mesh_category',
                'last_solver_tier',
                'last_retries_used',
                'last_repair_welded',
                'last_repair_fixed',
                'last_gate_status',
            ):
                if hasattr(settings, fname) and hasattr(snap, fname):
                    try:
                        setattr(settings, fname, getattr(snap, fname))
                    except (TypeError, ValueError):
                        pass  # type mismatch — silently ignore

        self.report(
            {'INFO'},
            f"QuadForge: {metrics.num_quads:,} quads "
            f"({metrics.quad_percentage:.0f}%), "
            f"{metrics.num_tris} tris, {elapsed:.2f}s"
        )
        print(f"[QuadForge] {'='*44}")
        return {'FINISHED'}


# ==========================================================
# QUADFORGE_OT_reset_defaults
# ==========================================================

class QUADFORGE_OT_reset_defaults(bpy.types.Operator):
    """Reset all QuadForge settings to their default values."""
    bl_idname     = "quadforge.reset_defaults"
    bl_label      = "Reset Defaults"
    bl_description = "Reset all QuadForge settings to defaults"

    def execute(self, context):
        s = context.scene.quadforge_settings
        s.preset                 = 'CUSTOM'
        s.target_quad_count      = 5000
        s.curvature_adaptivity   = 0.5
        s.exact_quad_count       = False
        s.auto_detect_hard_edges = True
        s.hard_edge_angle_deg    = 30.0
        s.use_normals            = False
        s.use_materials          = False
        s.use_vertex_colors      = False
        s.use_uv_seams           = False
        s.symmetry_x             = False
        s.symmetry_y             = False
        s.symmetry_z             = False
        s.smooth_iterations      = 10
        s.smooth_strength        = 0.5
        s.feature_snap_distance  = 0.1
        s.num_threads            = 0
        s.use_gpu                = False
        # Algorithm selection
        if hasattr(s, 'field_solver'):      s.field_solver      = 'KNOPPEL'
        if hasattr(s, 'param_method'):      s.param_method      = 'MIQ'
        if hasattr(s, 'extraction_method'): s.extraction_method = 'ISO'
        # Output options
        if hasattr(s, 'shade_smooth_output'): s.shade_smooth_output = True
        if hasattr(s, 'keep_original'):       s.keep_original       = True

        self.report({'INFO'}, "QuadForge: Settings reset to defaults")
        return {'FINISHED'}


# ==========================================================
# QUADFORGE_OT_paint_density
# ==========================================================

class QUADFORGE_OT_paint_density(bpy.types.Operator):
    """Activate Vertex Paint mode to paint a density map.

    Red = finer quads, Green = coarser quads (matches QuadRemesher convention).
    Ensures a colour attribute exists before switching modes.
    """
    bl_idname     = "quadforge.paint_density"
    bl_label      = "Paint Density Map"
    bl_description = (
        "Switch to Vertex Paint mode to paint a density map "
        "(red = finer quads, green = coarser quads)"
    )

    @classmethod
    def poll(cls, context):
        return (
            context.active_object is not None
            and context.active_object.type == 'MESH'
            and context.mode == 'OBJECT'
        )

    def execute(self, context):
        obj  = context.active_object
        mesh = obj.data

        # Ensure a colour attribute named 'QuadForge_Density' exists
        attr_name = "QuadForge_Density"
        if attr_name not in mesh.color_attributes:
            mesh.color_attributes.new(
                name=attr_name,
                type='FLOAT_COLOR',
                domain='POINT',
            )
            # BUG-J FIX (v117): initialise to neutral grey via foreach_set.
            # Was O(V) Python loop — each `d.color = ...` is a separate Python
            # attribute write; for 100K vertices that is ~100 ms of UI freeze.
            attr = mesh.color_attributes[attr_name]
            _grey = np.full(len(attr.data) * 4, 0.5, dtype=np.float32)
            _grey[3::4] = 1.0   # alpha channel → 1.0
            attr.data.foreach_set("color", _grey)

        # Set as active colour attribute
        mesh.color_attributes.active_color = mesh.color_attributes[attr_name]

        # Enable vertex colour display and switch to vertex paint
        obj.data.use_paint_mask = False
        bpy.ops.object.mode_set(mode='VERTEX_PAINT')

        self.report(
            {'INFO'},
            "QuadForge Density Map: red = finer, green = coarser. "
            "Return to Object Mode when done."
        )
        return {'FINISHED'}


# ==========================================================
# QUADFORGE_OT_visualize_field — Debug cross-field overlay
# ==========================================================

class QUADFORGE_OT_visualize_field(bpy.types.Operator):
    """Visualize the computed cross-field as short edge lines in the viewport.

    Creates a temporary mesh object showing per-face field directions as
    two perpendicular line segments (the four arms of the cross) at each
    face centroid. Useful for debugging field alignment before a full remesh.

    The visualization object is named '<source>_QF_Field' and can be
    deleted manually when no longer needed.
    """
    bl_idname      = "quadforge.visualize_field"
    bl_label       = "Visualize Cross-Field"
    bl_description = (
        "Create a debug mesh showing the computed cross-field directions "
        "as short line segments on the surface"
    )
    bl_options = {'REGISTER', 'UNDO'}

    scale: bpy.props.FloatProperty(
        name="Arrow Scale",
        description="Length of each field direction line relative to average edge length",
        default=0.3,
        min=0.05,
        max=2.0,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None
            and obj.type == 'MESH'
            and context.mode == 'OBJECT'
        )

    def execute(self, context):
        import numpy as np

        obj      = context.active_object
        settings = context.scene.quadforge_settings

        try:
            from .bridge import extract_mesh, build_params, QFParams
            from .engine.curvature import compute_curvature
            from .engine.field import compute_cross_field
            from .engine.features import extract_features
        except Exception as e:
            self.report({'ERROR'}, f"QuadForge: import error — {e}")
            return {'CANCELLED'}

        self.report({'INFO'}, "QuadForge: Computing cross-field for visualization…")

        try:
            input_mesh = extract_mesh(obj)
        except Exception as e:
            self.report({'ERROR'}, f"QuadForge: mesh extraction failed — {e}")
            return {'CANCELLED'}

        vertices  = np.asarray(input_mesh.vertices, dtype=np.float64)
        faces_arr = np.asarray(input_mesh.faces,    dtype=np.int32)

        # --- Feature edges (optional, best-effort) ---
        feature_edges = set()
        try:
            feats         = extract_features(obj, settings)
            feature_edges = feats.all_features
        except Exception:
            pass

        # --- Curvature directions ---
        try:
            curv = compute_curvature(vertices, faces_arr)
            curv_dirs = curv.dir1
        except Exception:
            curv_dirs = None

        # --- Cross-field ---
        try:
            cf = compute_cross_field(
                vertices, faces_arr,
                feature_edges=feature_edges or None,
                curvature_dirs=curv_dirs,
                use_eigensolver=(len(faces_arr) < 50_000),
            )
            field_dirs = cf.field_directions  # (F, 2) — two 3D directions per face
        except Exception as e:
            self.report({'ERROR'}, f"QuadForge: cross-field failed — {e}")
            return {'CANCELLED'}

        # --- Build visualization mesh (two line-quad per face centroid) ---
        avg_edge = _estimate_avg_edge_length(vertices, faces_arr)
        arm_len  = avg_edge * self.scale

        vis_verts = []
        vis_edges = []

        for fi, face in enumerate(faces_arr):
            centroid = np.mean(vertices[face], axis=0)

            if fi < len(field_dirs):
                d = field_dirs[fi]
                if d.ndim == 1:
                    # Single direction — build perpendicular
                    d0 = d / (np.linalg.norm(d) + 1e-15)
                    # Perpendicular in-plane: cross with face normal
                    fn = np.cross(vertices[face[1]] - vertices[face[0]],
                                  vertices[face[2]] - vertices[face[0]])
                    fn /= (np.linalg.norm(fn) + 1e-15)
                    d1 = np.cross(fn, d0)
                    d1 /= (np.linalg.norm(d1) + 1e-15)
                else:
                    d0 = d[0] / (np.linalg.norm(d[0]) + 1e-15)
                    d1 = d[1] / (np.linalg.norm(d[1]) + 1e-15)
            else:
                continue

            base = len(vis_verts)
            vis_verts.extend([
                centroid - d0 * arm_len,
                centroid + d0 * arm_len,
                centroid - d1 * arm_len,
                centroid + d1 * arm_len,
            ])
            vis_edges.extend([(base, base + 1), (base + 2, base + 3)])

        if not vis_verts:
            self.report({'WARNING'}, "QuadForge: No field data to visualize")
            return {'CANCELLED'}

        # Build and link the visualization mesh
        vis_name = f"{obj.name}_QF_Field"
        # Remove previous visualization if it exists
        old = bpy.data.objects.get(vis_name)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)

        vis_mesh = bpy.data.meshes.new(vis_name)
        vis_mesh.from_pydata(
            [v.tolist() for v in vis_verts],
            [list(e) for e in vis_edges],
            [],
        )
        vis_mesh.update()

        vis_obj = bpy.data.objects.new(vis_name, vis_mesh)
        vis_obj.matrix_world = obj.matrix_world.copy()
        context.collection.objects.link(vis_obj)

        # Tint the visualization object
        vis_obj.color = (0.1, 0.8, 1.0, 1.0)
        vis_obj.show_in_front = True

        self.report(
            {'INFO'},
            f"QuadForge: Field visualization created — {len(faces_arr)} face directions "
            f"({vis_name}). Delete when done."
        )
        return {'FINISHED'}


def _estimate_avg_edge_length(vertices: 'np.ndarray', faces: 'np.ndarray') -> float:
    """Estimate average edge length from a random sample of triangles."""
    import numpy as np
    if len(faces) == 0:
        return 0.05
    sample = min(500, len(faces))
    indices = np.random.choice(len(faces), sample, replace=False)
    lengths = []
    for fi in indices:
        tri = faces[fi]
        for i in range(3):
            e = vertices[tri[(i + 1) % 3]] - vertices[tri[i]]
            lengths.append(np.linalg.norm(e))
    return float(np.mean(lengths)) if lengths else 0.05




# ==========================================================
# QUADFORGE_OT_check_environment — v8 addition
# ==========================================================

class QUADFORGE_OT_check_environment(bpy.types.Operator):
    """Check QuadForge runtime environment and report to System Console.

    Runs all diagnostic checks (Blender version, NumPy, SciPy, native library)
    and prints a summary to the Blender System Console. Useful for bug reports
    and troubleshooting installation issues.
    """
    bl_idname     = "quadforge.check_environment"
    bl_label      = "Check Environment"
    bl_description = "Run QuadForge environment diagnostics and report to System Console"

    def execute(self, context):
        from .utils import get_environment_report, print_environment_info
        print_environment_info()
        report = get_environment_report()

        lines = []
        if not report["blender_ok"]:
            lines.append(f"UNSUPPORTED Blender: {report['blender_msg']}")
        if not report["numpy_ok"]:
            lines.append(f"MISSING NumPy: {report['numpy_msg']}")
        if not report["native_lib"]:
            lines.append("Native engine not found — using Python fallback (slower)")

        if lines:
            self.report({'WARNING'}, " | ".join(lines))
        else:
            self.report(
                {'INFO'},
                f"QuadForge: All checks passed — "
                f"{report['platform']}/{report['arch']}, "
                f"{report['cpu_count']} cores, "
                f"{'native engine' if report['native_lib'] else 'Python fallback'}"
            )
        return {'FINISHED'}

# ==========================================================
# QUADFORGE_OT_check_subdiv — Catmull-Clark compatibility check
# ==========================================================

class QUADFORGE_OT_check_subdiv(bpy.types.Operator):
    """Check whether the active quad mesh is compatible with Catmull-Clark
    subdivision.  Reports quad percentage, irregular vertex count, minimum
    quad angle, and Scaled Jacobian quality to the System Console.

    Use this after remeshing to verify the output is subdivision-ready
    before applying a Subdivision Surface modifier.
    """
    bl_idname      = "quadforge.check_subdiv"
    bl_label       = "Check Subdiv Compatibility"
    bl_description = (
        "Analyse the active mesh for Catmull-Clark subdivision compatibility "
        "and print a full report to the Blender System Console"
    )
    bl_options = {'REGISTER'}

    strict: bpy.props.BoolProperty(
        name="Strict Mode",
        description=(
            "Strict: require >99.5% quad faces and >99% regular vertices.\n"
            "Normal: allow up to 3% non-quad and 5% irregular (QuadForge targets)."
        ),
        default=False,
    )

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "QuadForge: No active mesh object selected.")
            return {'CANCELLED'}

        import numpy as np
        from .engine.subdiv import check_subdiv_compatibility

        mesh = obj.data
        mesh.calc_loop_triangles()

        # Extract vertices
        num_verts = len(mesh.vertices)
        verts_flat = np.empty(num_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", verts_flat)
        vertices = verts_flat.reshape((num_verts, 3))

        # Extract faces (polygons, not triangulated)
        faces = []
        for poly in mesh.polygons:
            faces.append(list(poly.vertices))

        if len(faces) == 0:
            self.report({'ERROR'}, "QuadForge: Mesh has no faces.")
            return {'CANCELLED'}

        # Run the check
        report = check_subdiv_compatibility(vertices, faces, strict=self.strict)

        # Print full report to System Console
        print(report.full_report())

        # Show summary in Blender header
        if report.is_compatible:
            self.report(
                {'INFO'},
                f"QuadForge CC-Check ✅  {report.summary()}"
            )
        else:
            self.report(
                {'WARNING'},
                f"QuadForge CC-Check ⚠️  {report.summary()} — See System Console for details."
            )

        return {'FINISHED'}




# ==========================================================
# QUADFORGE_OT_remesh_multiresolution — v2.0 Multi-Resolution
# ==========================================================

class QUADFORGE_OT_remesh_multiresolution(bpy.types.Operator):
    """Remesh using the v2.0 hierarchical coarse-to-fine pipeline.

    Optimal for meshes with > 200K triangles: decimates to a coarse mesh,
    solves the field globally, then prolongates and refines at full resolution.
    Typically 3–5× faster than the standard pipeline on large meshes.
    """

    bl_idname  = "quadforge.remesh_multiresolution"
    bl_label   = "⚡ Remesh (Multi-Res)"
    bl_description = (
        "Run the v2.0 multi-resolution pipeline — best for meshes > 200K triangles.\n"
        "Uses a coarse-to-fine strategy (QEM decimation → field solve → prolongation)"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (
            context.mode == 'OBJECT'
            and context.active_object is not None
            and context.active_object.type == 'MESH'
        )

    def execute(self, context):
        from .bridge import extract_mesh, build_params, build_output_mesh
        from .engine.multiresolution import run_multiresolution_pipeline
        from .engine.metrics import compute_quality_metrics

        obj = context.active_object
        settings = context.scene.quadforge_settings
        snap = _SettingsSnapshot(settings)
        # Force multi-res on regardless of property
        snap.use_multiresolution = True

        try:
            input_mesh = extract_mesh(obj)
        except Exception as e:
            self.report({'ERROR'}, f"QuadForge: Failed to extract mesh — {e}")
            return {'CANCELLED'}

        params = build_params(snap)

        # Progress output to header
        def _cb(stage, prog):
            pass  # synchronous operator — no modal timer needed

        self.report({'INFO'}, "QuadForge Multi-Res: processing…")

        try:
            result = run_multiresolution_pipeline(
                input_mesh, params,
                obj=None,
                settings=snap,
                progress_cb=_cb,
            )
        except Exception as e:
            import traceback as _tb
            _tb.print_exc()
            self.report({'ERROR'}, f"QuadForge Multi-Res failed: {e}")
            return {'CANCELLED'}

        # Support both _MultiResResult object and (verts, faces) tuple
        if hasattr(result, 'verts'):
            out_verts = result.verts
            out_faces = result.faces
            out_uvs   = getattr(result, 'output_uvs', None)
            out_norms = getattr(result, 'output_normals', None)
        else:
            out_verts, out_faces = result
            out_uvs = None
            out_norms = None

        if out_verts is None or len(out_faces) == 0:
            self.report({'ERROR'}, "QuadForge Multi-Res: empty output — try increasing target quad count")
            return {'CANCELLED'}

        new_mesh = build_output_mesh(
            name=f"{obj.name}_QuadForge",
            vertices=out_verts,
            faces=out_faces,
            normals=out_norms,
        )
        new_obj = bpy.data.objects.new(f"{obj.name}_QuadForge", new_mesh)
        context.collection.objects.link(new_obj)
        new_obj.matrix_world = obj.matrix_world.copy()
        for slot in obj.material_slots:
            if slot.material:
                new_obj.data.materials.append(slot.material)
        if out_uvs is not None and len(out_uvs) > 0:
            try:
                uv_layer = new_mesh.uv_layers.new(name="QuadForge_UV")
                # BUG-H FIX (v117): vectorized UV expansion (same fix as _build_output).
                _uv_arr  = np.asarray(out_uvs, dtype=np.float32)
                _vi_flat = np.concatenate([np.asarray(f, dtype=np.int32)
                                           for f in out_faces])
                _vi_flat = np.clip(_vi_flat, 0, len(_uv_arr) - 1)
                uv_layer.data.foreach_set("uv", _uv_arr[_vi_flat].ravel())
            except Exception as _e:
                print(f"[QuadForge] Multi-Res UV apply skipped: {_e}")
        # BUG-I FIX (v117): shade-smooth via foreach_set — was O(N) Python loop.
        if snap.shade_smooth_output:
            _sm = np.ones(len(new_mesh.polygons), dtype=bool)
            new_mesh.polygons.foreach_set("use_smooth", _sm)
        new_mesh.update()
        bpy.ops.object.select_all(action='DESELECT')
        new_obj.select_set(True)
        context.view_layer.objects.active = new_obj
        if not snap.keep_original:
            obj.hide_set(True)

        # Quality report
        # BUG-T FIX (v119): compute_quality_metrics() returns a QualityMetrics
        # DATACLASS, not a dict.  The previous code called metrics.get('key', default)
        # which raises AttributeError on every field access.  The surrounding
        # `except Exception: pass` silently swallowed all five errors, so
        # settings.last_* were NEVER updated after a Multi-Res remesh — the
        # N-panel "Last Run" block always showed stale values from the prior run.
        # Fix: access dataclass fields directly via attribute names.
        try:
            metrics = compute_quality_metrics(out_verts, out_faces)
            nq = metrics.num_quads                    # was: metrics.get('quad_count', ...)
            qp = metrics.quad_percentage              # was: metrics.get('quad_percentage', ...)
            self.report(
                {'INFO'},
                f"QuadForge Multi-Res: {nq} quads ({qp:.1f}% quad) → '{new_obj.name}'",
            )
            # Write back to result properties
            settings.last_quad_count      = nq
            settings.last_quad_percentage = qp
            settings.last_avg_valence     = metrics.avg_valence          # was: .get('avg_valence', ...)
            settings.last_min_jacobian    = metrics.min_scaled_jacobian  # was: .get('min_jacobian', ...)
            settings.last_mean_jacobian   = metrics.mean_scaled_jacobian # was: .get('mean_jacobian', ...)
            settings.last_input_object    = obj.name
        except Exception as _metrics_err:
            # Log to console so future bugs in this block are not silent
            print(f"[QuadForge] Multi-Res metrics write-back failed: {_metrics_err}")

        # BUG FIX v42: Dominance Layer report writeback (multi-res path).
        # The multi-res operator is synchronous but still passes a
        # _SettingsSnapshot into the pipeline rather than the live PG, so
        # any dominance report fields written by the pipeline end up on
        # the snapshot and never reach the N-panel. Mirror the threaded
        # path's writeback block so multi-res runs also surface the
        # confidence score, mesh category, retries, and repair counts.
        for fname in (
            'last_confidence_score',
            'last_mesh_category',
            'last_solver_tier',
            'last_retries_used',
            'last_repair_welded',
            'last_repair_fixed',
            'last_gate_status',
        ):
            if hasattr(settings, fname) and hasattr(snap, fname):
                try:
                    setattr(settings, fname, getattr(snap, fname))
                except (TypeError, ValueError):
                    pass

        return {'FINISHED'}


# ==========================================================
# v1.1 — Incremental Remeshing: Remesh Selection
# ==========================================================

class QUADFORGE_OT_remesh_selection(bpy.types.Operator):
    """Remesh only the selected faces while preserving the rest of the mesh.

    Select faces in Edit Mode, then run this operator.  The selected region
    is remeshed with the current QuadForge settings and stitched back to the
    unselected portion of the mesh by snapping boundary vertices.

    This implements the v1.1 Incremental Remeshing feature from the roadmap.
    """

    bl_idname  = "quadforge.remesh_selection"
    bl_label   = "Remesh Selection"
    bl_description = (
        "Remesh only the selected faces (Edit Mode). "
        "Preserves the unselected mesh and stitches at the boundary."
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None
            and obj.type == 'MESH'
            and context.mode == 'EDIT_MESH'
        )

    def execute(self, context):
        import bmesh
        from .engine.selection_remesh import (
            extract_selection,
            run_selection_pipeline,
            stitch_selection,
            build_selection_mask_from_face_indices,
        )
        from .bridge import build_params

        obj = context.active_object
        settings = context.scene.quadforge_settings

        # -- Switch to object mode to read the mesh safely --------------------
        bpy.ops.object.mode_set(mode='OBJECT')
        mesh = obj.data

        # -- Extract geometry arrays ------------------------------------------
        n_verts = len(mesh.vertices)
        n_faces = len(mesh.polygons)

        verts_flat = np.empty(n_verts * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", verts_flat)
        vertices = verts_flat.reshape(n_verts, 3)

        # We need the original polygon selection state — read from bmesh
        # (object-mode foreach_get doesn't give us 'select' per polygon reliably
        #  after mode switch, so we rebuild from bmesh)
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()

        selected_poly_indices = [f.index for f in bm.faces if f.select]
        bm.free()

        if len(selected_poly_indices) == 0:
            bpy.ops.object.mode_set(mode='EDIT')
            self.report({'WARNING'}, "QuadForge: No faces selected.")
            return {'CANCELLED'}

        if len(selected_poly_indices) == n_faces:
            bpy.ops.object.mode_set(mode='EDIT')
            self.report({'WARNING'},
                        "QuadForge: All faces selected — use regular Remesh instead.")
            return {'CANCELLED'}

        # -- Triangulate polygons (ear-clipping for any quads/n-gons) ---------
        faces_tri, tri_orig_face = _triangulate_mesh(mesh)
        selected_tri_mask = np.array(
            [tri_orig_face[i] in set(selected_poly_indices) for i in range(len(faces_tri))],
            dtype=bool,
        )

        # -- Extract selection ------------------------------------------------
        try:
            sel_data = extract_selection(
                vertices, faces_tri, selected_tri_mask,
            )
        except ValueError as e:
            bpy.ops.object.mode_set(mode='EDIT')
            self.report({'ERROR'}, f"QuadForge: {e}")
            return {'CANCELLED'}

        # -- Build params snapshot --------------------------------------------
        params = build_params(settings)

        # -- Run pipeline on the sub-mesh -------------------------------------
        self.report({'INFO'}, "QuadForge: Running selection remesh…")
        try:
            rem_verts, rem_faces = run_selection_pipeline(
                sel_data, params,
                progress_cb=lambda s, p: None,  # TODO: hook into timer
            )
        except Exception as e:
            bpy.ops.object.mode_set(mode='EDIT')
            self.report({'ERROR'}, f"QuadForge: Pipeline error — {e}")
            return {'CANCELLED'}

        # -- Stitch back ------------------------------------------------------
        try:
            result = stitch_selection(
                vertices, faces_tri, selected_tri_mask,
                rem_verts, rem_faces,
                sel_data,
            )
        except Exception as e:
            bpy.ops.object.mode_set(mode='EDIT')
            self.report({'ERROR'}, f"QuadForge: Stitch error — {e}")
            return {'CANCELLED'}

        # -- Apply the stitched mesh back to the Blender object ---------------
        from .bridge import build_output_mesh
        new_mesh = build_output_mesh(
            name=mesh.name,
            vertices=result.vertices.astype(np.float32),
            faces=result.faces,
        )
        obj.data = new_mesh
        bpy.data.meshes.remove(mesh)

        # BUG-I FIX (v117): shade-smooth via foreach_set — was O(N) Python loop.
        if settings.shade_smooth_output:
            smooth_arr = np.ones(len(new_mesh.polygons), dtype=bool)
            new_mesh.polygons.foreach_set("use_smooth", smooth_arr)
        new_mesh.update()

        bpy.ops.object.mode_set(mode='EDIT')

        self.report(
            {'INFO'},
            f"QuadForge: Selection remesh done — "
            f"{len(result.faces)} faces "
            f"({result.boundary_snapped}/{result.boundary_total} boundary verts snapped).",
        )
        return {'FINISHED'}


def _triangulate_mesh(mesh) -> tuple:
    """Return (triangulated_faces, tri_to_orig_face_idx) from a Blender mesh.

    Quads and n-gons are ear-clipped into triangles.  The second return value
    maps each output triangle index to its source polygon index.
    """
    tris  = []
    tri_orig = []
    for fi, poly in enumerate(mesh.polygons):
        verts = list(poly.vertices)
        # Fan triangulation from first vertex
        for i in range(1, len(verts) - 1):
            tris.append([verts[0], verts[i], verts[i + 1]])
            tri_orig.append(fi)
    return np.array(tris, dtype=np.int32), tri_orig


# ==========================================================
# v1.4 — Live Preview Mode
# ==========================================================

class QUADFORGE_OT_live_preview(bpy.types.Operator):
    """Show a real-time wireframe cross-field overlay in the viewport.

    Runs only the curvature + field stages (< 0.5 s on medium meshes) and
    displays the result as a temporary arrow mesh object named
    '_QF_Preview_Field'.  Run again to refresh; delete the object manually
    to remove the overlay.

    This implements the v1.4 Live Preview Mode from the roadmap.
    """

    bl_idname  = "quadforge.live_preview"
    bl_label   = "Preview Field"
    bl_description = (
        "Display a cross-field direction overlay in the viewport "
        "(runs fast — no full remesh)."
    )
    bl_options = {'REGISTER', 'UNDO'}

    max_arrows: bpy.props.IntProperty(
        name="Max Arrows",
        description="Maximum number of field direction arrows to display",
        default=2000,
        min=100,
        max=20000,
    )

    fast: bpy.props.BoolProperty(
        name="Fast Mode",
        description=(
            "Use fast curvature-only field (instant) instead of the "
            "full Knöppel eigensolver"
        ),
        default=True,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and context.mode == 'OBJECT'

    def execute(self, context):
        from .engine.live_preview import compute_field_preview
        from .bridge import build_params

        obj      = context.active_object
        settings = context.scene.quadforge_settings
        mesh     = obj.data

        # Extract geometry
        n_verts = len(mesh.vertices)
        n_faces = len(mesh.polygons)

        verts_flat = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", verts_flat)
        vertices = verts_flat.reshape(n_verts, 3).astype(np.float64)

        faces_tri, _ = _triangulate_mesh(mesh)

        params = build_params(settings)

        try:
            preview = compute_field_preview(
                vertices, faces_tri, params,
                n_arrows=self.max_arrows,
                fast=self.fast,
            )
        except Exception as e:
            self.report({'ERROR'}, f"QuadForge Preview: {e}")
            return {'CANCELLED'}

        preview_verts, preview_edges = preview.to_edge_mesh()

        # Remove any previous preview object
        PREVIEW_NAME = "_QF_Preview_Field"
        if PREVIEW_NAME in bpy.data.objects:
            old = bpy.data.objects[PREVIEW_NAME]
            bpy.data.meshes.remove(old.data, do_unlink=True)

        # Build preview mesh
        pm = bpy.data.meshes.new(PREVIEW_NAME)
        pm.from_pydata(preview_verts, preview_edges, [])
        pm.update()

        po = bpy.data.objects.new(PREVIEW_NAME, pm)
        context.collection.objects.link(po)
        po.matrix_world = obj.matrix_world.copy()
        # Make it non-selectable and display in a distinct colour
        po.hide_select = True

        self.report(
            {'INFO'},
            f"QuadForge Preview: {len(preview_verts)//4} arrows  "
            f"({preview.elapsed*1000:.0f} ms)",
        )
        return {'FINISHED'}


# ==========================================================
# QUADFORGE_OT_detect_gpu — v1.2 GPU detection (operator)
# ==========================================================

class QUADFORGE_OT_detect_gpu(bpy.types.Operator):
    """Probe the system for a compatible GPU solver backend and
display the result in the Info header bar."""

    bl_idname  = "quadforge.detect_gpu"
    bl_label   = "Detect GPU"
    bl_description = (
        "Probe for CUDA / Metal / OpenCL GPU and select the best "
        "available backend for the sparse solver"
    )
    bl_options = {'REGISTER'}

    def execute(self, context):
        from .engine.gpu_solver import detect_gpu, reset_cache, set_backend
        settings = context.scene.quadforge_settings

        # Force re-probe
        reset_cache()
        info = detect_gpu()

        # Apply detected backend to settings
        if info.available:
            backend_map = {
                "cuda":   "CUDA",
                "metal":  "METAL",
                "opencl": "OPENCL",
                "cpu":    "CPU",
            }
            mapped = backend_map.get(info.backend, "AUTO")
            if settings.gpu_backend == "AUTO":
                pass  # keep AUTO — auto-detect will use the cached result
            else:
                settings.gpu_backend = mapped

            self.report(
                {'INFO'},
                f"QuadForge GPU: {info.device_name} [{info.backend.upper()}]  "
                f"~{info.estimated_speedup:.1f}× speedup"
            )
        else:
            settings.use_gpu = False
            self.report(
                {'WARNING'},
                "QuadForge GPU: No compatible GPU found — CPU solver will be used"
            )

        # Print full info to system console
        print("[QuadForge] GPU Detection Result:")
        print(f"  Available     : {info.available}")
        print(f"  Backend       : {info.backend}")
        print(f"  Device        : {info.device_name}")
        print(f"  VRAM          : {info.vram_mb} MB")
        print(f"  Driver        : {info.driver_version}")
        print(f"  Est. Speedup  : {info.estimated_speedup:.1f}×")
        for note in info.notes:
            print(f"  Note          : {note}")

        return {'FINISHED'}


# ==========================================================
# QUADFORGE_OT_apply_preset — one-click preset icon button
# ==========================================================

class QUADFORGE_OT_apply_preset(bpy.types.Operator):
    """Apply a QuadForge preset (used by the icon shortcut buttons in the N-panel)."""

    bl_idname  = "quadforge.apply_preset"
    bl_label   = "Apply Preset"
    bl_description = "Apply a named QuadForge preset"
    bl_options = {'INTERNAL', 'UNDO'}

    preset: bpy.props.EnumProperty(
        name="Preset",
        items=[
            ('ORGANIC',      "Organic",      ""),
            ('HARD_SURFACE', "Hard Surface", ""),
            ('SCULPT',       "Sculpt",       ""),
            ('ARCHITECTURE', "Architecture", ""),
            ('FAST',         "Fast",         ""),
            ('CUSTOM',       "Custom",       ""),
        ],
        default='ORGANIC',
    )

    def execute(self, context):
        settings = context.scene.quadforge_settings
        settings.preset = self.preset   # triggers _on_preset_changed callback
        self.report({'INFO'}, f"QuadForge: Applied preset '{self.preset}'")
        return {'FINISHED'}


# ==========================================================
# Registration + Keymap (final)
# ==========================================================

classes = (
    QUADFORGE_OT_remesh,
    QUADFORGE_OT_reset_defaults,
    QUADFORGE_OT_paint_density,
    QUADFORGE_OT_visualize_field,
    QUADFORGE_OT_check_environment,
    QUADFORGE_OT_check_subdiv,
    QUADFORGE_OT_remesh_selection,       # v1.1 Incremental Remeshing
    QUADFORGE_OT_live_preview,           # v1.4 Live Preview
    QUADFORGE_OT_detect_gpu,             # v1.2 GPU Detection
    QUADFORGE_OT_remesh_multiresolution, # v2.0 Multi-Resolution
    QUADFORGE_OT_apply_preset,           # v15.0 Preset icon buttons
)

addon_keymaps = []


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    wm = bpy.context.window_manager
    if wm.keyconfigs.addon:
        km = wm.keyconfigs.addon.keymaps.new(name='3D View', space_type='VIEW_3D')
        kmi = km.keymap_items.new("quadforge.remesh", 'R', 'PRESS', ctrl=True, alt=True)
        addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
