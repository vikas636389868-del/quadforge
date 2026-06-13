"""QuadForge preset system.

Five built-in presets override multiple parameters at once.
Each preset now includes algorithm-selection fields so the
user always gets a fully-configured engine, not just aesthetic tweaks.
"""

from __future__ import annotations

PRESETS = {
    'ORGANIC': {
        # Sizing
        'curvature_adaptivity': 0.8,
        # Features
        'auto_detect_hard_edges': False,
        'hard_edge_angle_deg': 60.0,
        'use_normals': False,
        'use_materials': False,
        # Smoothing
        'smooth_iterations': 15,
        'smooth_strength': 0.5,
        'feature_snap_distance': 0.05,
        # Algorithms — Knöppel field + MIQ + iso-line: best quality for smooth meshes
        'field_solver': 'KNOPPEL',
        'param_method': 'MIQ',
        'extraction_method': 'ISO',
        # Output
        'shade_smooth_output': True,
    },
    'HARD_SURFACE': {
        'curvature_adaptivity': 0.3,
        'auto_detect_hard_edges': True,
        'hard_edge_angle_deg': 25.0,
        'use_normals': True,
        'use_materials': True,
        'smooth_iterations': 5,
        'smooth_strength': 0.3,
        'feature_snap_distance': 0.2,
        # Hard surface: Knöppel for clean crease alignment, MIQ for crisp edge loops,
        # MOTORCYCLE for T-junction-free corners at feature edges
        'field_solver': 'KNOPPEL',
        'param_method': 'MIQ',
        'extraction_method': 'MOTORCYCLE',
        'shade_smooth_output': False,
    },
    'SCULPT': {
        'curvature_adaptivity': 0.7,
        'auto_detect_hard_edges': False,
        'hard_edge_angle_deg': 45.0,
        'use_normals': False,
        'use_materials': False,
        'smooth_iterations': 10,
        'smooth_strength': 0.5,
        'feature_snap_distance': 0.05,
        # Sculpt: Knöppel + Poisson for speed (sculpted meshes are smooth)
        'field_solver': 'KNOPPEL',
        'param_method': 'POISSON',
        'extraction_method': 'ISO',
        'shade_smooth_output': True,
    },
    'ARCHITECTURE': {
        'curvature_adaptivity': 0.1,
        'auto_detect_hard_edges': True,
        'hard_edge_angle_deg': 15.0,
        'use_normals': True,
        'use_materials': True,
        'exact_quad_count': True,
        'smooth_iterations': 3,
        'smooth_strength': 0.2,
        'feature_snap_distance': 0.3,
        # Architecture: IGM for strictest rectangular grid, Knöppel for alignment
        # MOTORCYCLE extraction for cleanest T-junction-free corners
        'field_solver': 'KNOPPEL',
        'param_method': 'IGM',
        'extraction_method': 'MOTORCYCLE',
        'shade_smooth_output': False,
    },
    'FAST': {
        'curvature_adaptivity': 0.2,
        'auto_detect_hard_edges': True,
        'hard_edge_angle_deg': 30.0,
        'use_normals': False,
        'use_materials': False,
        'smooth_iterations': 3,
        'smooth_strength': 0.4,
        'feature_snap_distance': 0.1,
        # Fast: curvature field + Poisson + greedy merge — maximum speed
        'field_solver': 'CURVATURE',
        'param_method': 'POISSON',
        'extraction_method': 'GREEDY',
        'shade_smooth_output': True,
    },
}


def apply_preset(settings, preset_name: str):
    """Apply a preset's values to a QFSettingsPropertyGroup.

    Parameters
    ----------
    settings : QFSettingsPropertyGroup
    preset_name : 'ORGANIC' | 'HARD_SURFACE' | 'SCULPT' | 'ARCHITECTURE' | 'FAST' | 'CUSTOM'
    """
    if preset_name == 'CUSTOM' or preset_name not in PRESETS:
        return

    for key, value in PRESETS[preset_name].items():
        if hasattr(settings, key):
            setattr(settings, key, value)
