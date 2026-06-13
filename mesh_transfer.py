"""QuadForge mesh transfer utilities.

Standalone module for converting between Blender mesh data and
the flat numpy arrays used by the QuadForge engine.

This is a thin convenience wrapper around bridge.extract_mesh()
and bridge.build_output_mesh() that exposes functions matching
the naming convention in the roadmap's architecture diagram.

Usage (from Blender Python):
    from QuadForge.mesh_transfer import blender_mesh_to_arrays, arrays_to_blender_mesh

    input_mesh = blender_mesh_to_arrays(context.active_object)
    # ... run engine ...
    new_obj = arrays_to_blender_mesh(context, verts, faces, name="MyMesh")
"""

from __future__ import annotations

import bpy
import numpy as np

from typing import Optional, List
from .bridge import extract_mesh, build_output_mesh, QFInputMesh

from .bridge import extract_mesh, build_output_mesh, QFInputMesh


def blender_mesh_to_arrays(obj: bpy.types.Object) -> QFInputMesh:
    """Extract a Blender mesh object into numpy arrays.

    Thin wrapper around bridge.extract_mesh() matching the naming
    convention described in the roadmap's mesh_transfer.py module.

    Parameters
    ----------
    obj : bpy.types.Object
        A Blender mesh object (obj.type == 'MESH').

    Returns
    -------
    QFInputMesh
        Container of numpy arrays: vertices (V×3), faces (F×3 triangulated),
        normals, uv_coords, vertex_colors, material_ids, original_faces.

    Raises
    ------
    ValueError
        If the object has no mesh data or zero geometry.
    TypeError
        If the object is not a mesh type.
    """
    if obj.type != 'MESH':
        raise TypeError(f"Expected a MESH object, got '{obj.type}'")
    return extract_mesh(obj)


def arrays_to_blender_mesh(
    context: bpy.types.Context,
    vertices: np.ndarray,
    faces: List[List[int]],
    name: str = "QuadForge_Result",
    source_obj: Optional[bpy.types.Object] = None,
    shade_smooth: bool = True,
) -> bpy.types.Object:
    """Create a new Blender object from numpy vertex/face arrays.

    Thin wrapper around bridge.build_output_mesh() that handles
    object creation, collection linking, and world matrix copy.

    Parameters
    ----------
    context : bpy.types.Context
    vertices : np.ndarray  shape (V, 3) float32/float64
    faces : list of lists of int  (quads and tris)
    name : str  name for the new mesh and object
    source_obj : bpy.types.Object, optional
        If provided, the world transform and materials are copied from it.
    shade_smooth : bool
        If True, apply smooth shading to all polygons.

    Returns
    -------
    bpy.types.Object
        The newly created and linked Blender object.
    """
    mesh = build_output_mesh(
        name=name,
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=faces,
    )

    # Apply shading
    for poly in mesh.polygons:
        poly.use_smooth = shade_smooth
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    context.collection.objects.link(obj)

    if source_obj is not None:
        obj.matrix_world = source_obj.matrix_world.copy()
        for slot in source_obj.material_slots:
            if slot.material:
                obj.data.materials.append(slot.material)

    return obj


def get_mesh_stats(obj: bpy.types.Object) -> dict:
    """Return basic statistics about a Blender mesh object.

    Useful for pre-remesh sanity checks.

    Returns dict with keys: vertices, edges, faces, is_manifold,
    has_uvs, has_vertex_colors, has_materials.
    """
    mesh = obj.data
    return {
        "vertices":          len(mesh.vertices),
        "edges":             len(mesh.edges),
        "faces":             len(mesh.polygons),
        "is_manifold":       all(e.is_manifold for e in mesh.edges),
        "has_uvs":           mesh.uv_layers.active is not None,
        "has_vertex_colors": bool(mesh.color_attributes),
        "has_materials":     len(obj.material_slots) > 0,
    }
