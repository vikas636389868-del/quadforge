import sys
import os
from unittest.mock import MagicMock

# 1. Mocking bpy aur uske sub-modules
class MockBpy(MagicMock):
    def __getattr__(self, name):
        return MagicMock()

sys.modules["bpy"] = MockBpy()
sys.modules["bpy.props"] = MockBpy()
sys.modules["bpy.types"] = MockBpy()
sys.modules["bpy.utils"] = MockBpy()
sys.modules["bmesh"] = MagicMock()
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
