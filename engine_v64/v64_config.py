"""Shared switches for the UNHCR feature and projection paths."""
from ortho_config import V128_K1, V128_K2, V128_K3
import os
V132_K1 = os.environ.get('SB_V132_K1', '0') == '1'
V132_K2 = os.environ.get('SB_V132_K2', '0') == '1'
V136_K1 = os.environ.get('SB_V136_K1', '0') == '1'
