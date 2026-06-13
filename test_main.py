import sys
from unittest.mock import MagicMock

# Blender ke 'bpy' module ko fake (mock) object se replace karein
# Isse pytest ko lagega ki bpy module exist karta hai
sys.modules["bpy"] = MagicMock()

def test_setup_working():
    # Ab yahan aap apni main file ko import kar sakte hain
    import bridge 
    assert True
