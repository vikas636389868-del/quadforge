
bl_info = {
    "name": "QuadForge",
    "author": "Reynold",
    "version": (84, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > QuadForge",
    "description": "Next-generation auto quad remeshing — faster than QuadRemesher (Dominance Layer)",
    "warning": "",
    "doc_url": "https://github.com/reynold/quadforge/blob/main/docs/USER_GUIDE.md",
    "category": "Mesh",
}

import importlib
import os
import sys

# ---------------------------------------------------------------------------
# Custom icon registry
# ---------------------------------------------------------------------------
# Populated in register(), cleared in unregister().
# Access from any module: from QuadForge import icon_id
_preview_collections: dict = {}


def icon_id(name: str) -> int:
    """Return the integer icon_id for a custom icon, or 0 if not loaded.

    Usage in panels::

        icon = icon_id("organic")
        row.operator("quadforge.remesh", icon_value=icon)
    """
    pcoll = _preview_collections.get("quadforge")
    if pcoll and name in pcoll:
        return pcoll[name].icon_id
    return 0


def _load_icons() -> None:
    try:
        import bpy.utils.previews  # noqa: PLC0415
    except ImportError:
        return  # Headless / test environment — silently skip

    pcoll = bpy.utils.previews.new()  # type: ignore[attr-defined]
    icons_dir = os.path.join(os.path.dirname(__file__), "icons")

    # Main logo
    logo_path = os.path.join(icons_dir, "quadforge_logo.png")
    if os.path.isfile(logo_path):
        try:
            pcoll.load("logo", logo_path, "IMAGE")
        except Exception:
            pass

    # Preset icons
    preset_dir = os.path.join(icons_dir, "preset_icons")
    for preset in ("organic", "hard_surface", "sculpt", "architecture", "fast"):
        path = os.path.join(preset_dir, f"{preset}.png")
        if os.path.isfile(path):
            try:
                pcoll.load(preset, path, "IMAGE")
            except Exception:
                pass  # duplicate key on F8 reload — ignore

    _preview_collections["quadforge"] = pcoll


def _unload_icons() -> None:
  
    try:
        import bpy.utils.previews  # noqa: PLC0415
    except ImportError:
        return
    for pcoll in _preview_collections.values():
        bpy.utils.previews.remove(pcoll)  # type: ignore[attr-defined]
    _preview_collections.clear()


def _reload_submodules():
    module_names = [
        # Engine: pure-Python geometry modules (no bpy dependency)
        "QuadForge.engine.halfedge",
        "QuadForge.engine.topology",
        "QuadForge.engine.spatial",
        "QuadForge.engine.sizing",
        "QuadForge.engine.curvature",
        "QuadForge.engine.features",
        "QuadForge.engine.constraints",
        "QuadForge.engine.smoothing",
        "QuadForge.engine.field",
        "QuadForge.engine.field_smoothing",
        "QuadForge.engine.sparse_solver",
        "QuadForge.engine.combing",
        "QuadForge.engine.parametrize",
        "QuadForge.engine.extraction",
        "QuadForge.engine.quad_merger",
        "QuadForge.engine.feature_protection",
        "QuadForge.engine.motorcycle",
        "QuadForge.engine.singularity",
        "QuadForge.engine.symmetry",
        "QuadForge.engine.exact_count",
        "QuadForge.engine.output_enhance",
        "QuadForge.engine.projection",
        "QuadForge.engine.snapping",
        "QuadForge.engine.transfer",
        "QuadForge.engine.metrics",
        "QuadForge.engine.subdiv",
        "QuadForge.engine.selection_remesh",
        "QuadForge.engine.neural_singularity",
        "QuadForge.engine.multiresolution",
        "QuadForge.engine.live_preview",
        "QuadForge.engine.gpu_solver",
        # v4.0 — Dominance Layer (Roadmap Part XI + XIII).
        # These MUST be reloaded before pipeline/dominance_pipeline so that
        # F8 "Reload Scripts" picks up edits in any of them.
        "QuadForge.engine.determinism",
        "QuadForge.engine.mesh_repair",
        "QuadForge.engine.mesh_classifier",
        "QuadForge.engine.artist_intelligence",
        "QuadForge.engine.quality_optimizer",
        "QuadForge.engine.confidence",
        "QuadForge.engine.edge_flow_refine",
        "QuadForge.engine.progressive_preview",
        "QuadForge.engine.dominance_pipeline",
        "QuadForge.engine.pipeline",
        "QuadForge.engine",
        # Addon: Blender-facing modules
        "QuadForge.bridge",
        "QuadForge.presets",
        "QuadForge.callbacks",
        "QuadForge.properties",
        "QuadForge.preferences",
        "QuadForge.operators",
        "QuadForge.panels",
    ]
    for name in module_names:
        if name in sys.modules:
            importlib.reload(sys.modules[name])


if "QuadForge.bridge" in sys.modules:
    _reload_submodules()

from . import (  # noqa: E402
    bridge,
    presets,
    callbacks,
    properties,
    preferences,
    operators,
    panels,
    utils,
)
from .engine import (  # noqa: E402, F401
    halfedge,
    topology,
    spatial,
    sizing,
    curvature,
    features,
    constraints,
    smoothing,
    field,
    field_smoothing,
    sparse_solver,
    combing,
    parametrize,
    extraction,
    quad_merger,
    feature_protection,
    motorcycle,
    singularity,
    symmetry,
    exact_count,
    output_enhance,
    projection,
    snapping,
    transfer,
    metrics,
    subdiv,
    selection_remesh,
    neural_singularity,
    multiresolution,
    live_preview,
    gpu_solver,
    pipeline,
)


def register():
    _load_icons()
    preferences.register()
    properties.register()
    operators.register()
    panels.register()
    # Print environment diagnostics to Blender System Console on load
    try:
        utils.print_environment_info()
    except Exception:
        pass


def unregister():
    panels.unregister()
    operators.unregister()
    properties.unregister()
    preferences.unregister()
    _unload_icons()


if __name__ == "__main__":
    register()

import sys
from unittest.mock import MagicMock

# 1. Blender environment mock check (Only if not in Blender)
try:
    import bpy
except ImportError:
    # Ye block sirf non-Blender environment (pytest) ke liye hai
    sys.modules["bpy"] = MagicMock()
    # Baaki modules ko bhi mock karein
    sys.modules["bpy.props"] = MagicMock()
    sys.modules["bpy.types"] = MagicMock()
    sys.modules["bpy.utils"] = MagicMock()
    sys.modules["bmesh"] = MagicMock()

# 2. Package imports (Safe mode mein)
try:
    from . import bridge, operators, panels, mesh_transfer
except Exception as e:
    print(f"Skipping imports due to: {e}")
