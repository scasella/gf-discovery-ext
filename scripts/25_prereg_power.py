"""Label-free, DATA-FREE null calibration for the prereg: what a geometrically
perfect ring would score under the uniform-angle closure null, for the survivor
patterns the pool table implies. No cache reads, no SAE values, no model.
"""
import sys, numpy as np
from pathlib import Path
ROOT = Path("/Users/scasella/Downloads/gf-discovery")
sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("g13", ROOT / "scripts/13_rerun_gate.py")
g13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g13)

cases = [
    ("k=4 full ring (seasons; cardinals-only compass as k=4)", [0,1,2,3], 4),
    ("k=5 full ring", [0,1,2,3,4], 5),
    ("k=6 full ring", [0,1,2,3,4,5], 6),
    ("k=8 full ring (compass, all 8)", list(range(8)), 8),
    ("k=8, cardinals only (4 of 8, alternating)", [0,2,4,6], 8),
    ("k=8, one dropout", [0,1,2,3,4,5,6], 8),
    ("k=8, two adjacent dropouts", [0,1,2,3,4,5], 8),
    ("k=8, two non-adjacent dropouts", [0,1,2,3,4,6], 8),
    ("k=12 full ring (zodiac / keys)", list(range(12)), 12),
    ("k=12 naturals in FIFTHS order (7 of 12: 0-5,11)", [0,1,2,3,4,5,11], 12),
    ("k=12 naturals in CHROMATIC order (7 of 12: 0,2,4,5,7,9,11)", [0,2,4,5,7,9,11], 12),
    ("k=13 full (kinship, for reference only)", list(range(13)), 13),
]
print(f"{'case':<58} {'n':>3} {'gap_deg':>8} {'ref_p':>9} {'run':>4} {'testable':>9}")
for name, keep, full_k in cases:
    r = g13.reference_ring(keep, full_k, seed=0)
    print(f"{name:<58} {len(keep):>3} {r['ref_gap_deg']:>8.2f} {r['ref_p_closed']:>9.5f} "
          f"{r['missing_run']:>4} {str(r['ref_p_closed'] < 0.05):>9}")
