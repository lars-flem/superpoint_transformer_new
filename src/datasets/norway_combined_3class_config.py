import numpy as np

from src.datasets.norway_combined_tiles import TILES, TILES_MINI, REGIONS

__all__ = [
    "TILES",
    "TILES_MINI",
    "REGIONS",
    "ID2TRAINID",
    "CLASS_NAMES",
    "CLASS_COLORS",
    "NORWAY_COMBINED_3CLASS_NUM_CLASSES",
]

# Mapping from LAS classification (0-255) -> train id, mirroring the viken2022
# 3-class setup (ground / not_ground / building) extended to cover the extra
# classes seen in Oslo, Bergen2020, and Bergen2022.
#
# Train ids:
#   0 = ground, 1 = not_ground, 2 = building, 3 = ignored
#
# Anything not explicitly mapped defaults to ignored.
ID2TRAINID = np.full(256, 3, dtype=np.int64)  # default: ignored

# Ground
ID2TRAINID[2] = 0    # Terreng (Ground)
ID2TRAINID[11] = 0   # Vegbane (Road, Bergen2022)

# Not ground
ID2TRAINID[3] = 1    # Lav vegetasjon
ID2TRAINID[4] = 1    # Medium vegetasjon
ID2TRAINID[5] = 1    # Høy vegetasjon
ID2TRAINID[13] = 1   # Ledning beskyttelse (Powerline protection, Oslo/Bergen2022)
ID2TRAINID[14] = 1   # Ledning / Powerlines
ID2TRAINID[15] = 1   # Mast / Transmission tower
ID2TRAINID[16] = 1   # Wire-structure connector (Oslo)
ID2TRAINID[17] = 1   # Bro (Bridge)
ID2TRAINID[19] = 1   # Overhead structure (Oslo)
ID2TRAINID[23] = 1   # Gress (Grass, Viken)
ID2TRAINID[64] = 1   # Veldig lav vegetasjon (Very low veg, Oslo)

# Building
ID2TRAINID[6] = 2    # Bygning

# Explicitly ignored (already covered by default; listed for clarity)
ID2TRAINID[1] = 3    # Uklassifisert
ID2TRAINID[7] = 3    # Støy (Noise / low point)
ID2TRAINID[9] = 3    # Vann (Water)
ID2TRAINID[22] = 3   # Temporal Exclusion (Oslo)
ID2TRAINID[24] = 3   # Snø (Snow)

CLASS_NAMES = ["ground", "not_ground", "building", "ignored"]
CLASS_COLORS = [
    [140, 90, 60],    # ground: brown
    [180, 180, 180],  # not_ground: grey
    [255, 210, 0],    # building: yellow
    [0, 0, 0],        # ignored: black
]

NORWAY_COMBINED_3CLASS_NUM_CLASSES = 3
