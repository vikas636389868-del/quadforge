"""QuadForge addon preferences.

Accessible via Edit > Preferences > Add-ons > QuadForge.
"""

import bpy
from bpy.props import IntProperty, BoolProperty, EnumProperty


class QUADFORGEPreferences(bpy.types.AddonPreferences):
    bl_idname = "QuadForge"

    default_thread_count: IntProperty(
        name="Default Thread Count",
        description="Default number of threads used for parallel stages (0 = auto-detect all cores)",
        default=0,
        min=0,
        max=64,
    )

    show_debug_output: BoolProperty(
        name="Show Debug Output",
        description="Print detailed pipeline debug info to the Blender System Console",
        default=False,
    )

    auto_hide_input: BoolProperty(
        name="Auto-Hide Input Object",
        description=(
            "Automatically hide the input mesh after a successful remesh "
            "(overridden by the 'Keep Original' toggle in the panel)"
        ),
        default=False,
    )

    default_field_solver: EnumProperty(
        name="Default Field Solver",
        description="Global default for the cross-field solver algorithm",
        items=[
            ('KNOPPEL',   "Knöppel 2013",  "Globally optimal eigensolver (best quality)"),
            ('CURVATURE', "Curvature Only", "Fast curvature-aligned field"),
        ],
        default='KNOPPEL',
    )

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text="Performance", icon='MOD_SMOOTH')
        box.prop(self, "default_thread_count")
        box.prop(self, "default_field_solver")

        box2 = layout.box()
        box2.label(text="Behaviour", icon='PREFERENCES')
        box2.prop(self, "auto_hide_input")
        box2.prop(self, "show_debug_output")

        layout.separator()
        layout.operator(
            "quadforge.check_environment",
            text="Check Environment",
            icon='CONSOLE',
        )


classes = (QUADFORGEPreferences,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
