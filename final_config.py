"""Final-phase experiment switches. Defaults retain the complete  recipe."""
import os
_phase = 'dev'
def enabled(name):
    return os.environ.get('SB_V157_' + name, '1').lower() not in ('0', 'false', 'off')
def set_phase(phase):
    global _phase
    _phase = phase
def is_final():
    return _phase == 'final'
