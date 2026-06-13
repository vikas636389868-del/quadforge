"""QuadForge property update callbacks.

Thin module kept for API compatibility — the main callback
(_on_preset_changed) now lives in properties.py to avoid a
circular import at registration time.
"""

from __future__ import annotations


def on_preset_changed(self, context):
    """Called when the preset dropdown value changes.

    Delegates to presets.apply_preset() which does the actual work.
    This alias is kept for backward compatibility with any external
    code that referenced QuadForge.callbacks.on_preset_changed.
    """
    from .presets import apply_preset
    apply_preset(self, self.preset)
