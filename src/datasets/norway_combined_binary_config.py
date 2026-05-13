import numpy as np

from src.datasets.norway_combined_tiles import TILES, TILES_MINI, REGIONS

__all__ = [
    "TILES",
    "TILES_MINI",
    "REGIONS",
    "ID2TRAINID",
    "CLASS_NAMES",
    "CLASS_COLORS",
    "NORWAY_COMBINED_BINARY_NUM_CLASSES",
]

# Mapping from LAS classification (0-255) -> train id, binary version of the
# combined Norway setup. Same as the 3-class variant but Building (LAS 6) is
# folded into not_ground.
#
# Train ids:
#   0 = ground, 1 = not_ground, 2 = ignored
ID2TRAINID = np.full(256, 2, dtype=np.int64)  # default: ignored

# Ground
ID2TRAINID[2] = 0    # Terreng (Ground)
ID2TRAINID[11] = 0   # Vegbane (Road, Bergen2022)

# Not ground (incl. building)
ID2TRAINID[3] = 1    # Lav vegetasjon
ID2TRAINID[4] = 1    # Medium vegetasjon
ID2TRAINID[5] = 1    # Høy vegetasjon
ID2TRAINID[6] = 1    # Bygning (folded into not_ground for binary)
ID2TRAINID[13] = 1   # Ledning beskyttelse
ID2TRAINID[14] = 1   # Ledning / Powerlines
ID2TRAINID[15] = 1   # Mast / Transmission tower
ID2TRAINID[16] = 1   # Wire-structure connector (Oslo)
ID2TRAINID[17] = 1   # Bro (Bridge)
ID2TRAINID[19] = 1   # Overhead structure (Oslo)
ID2TRAINID[23] = 1   # Gress (Grass, Viken)
ID2TRAINID[64] = 1   # Veldig lav vegetasjon (Oslo)

# Explicitly ignored
ID2TRAINID[1] = 2    # Uklassifisert
ID2TRAINID[7] = 2    # Støy
ID2TRAINID[9] = 2    # Vann
ID2TRAINID[22] = 2   # Temporal Exclusion (Oslo)
ID2TRAINID[24] = 2   # Snø

CLASS_NAMES = ["ground", "not_ground", "ignored"]
CLASS_COLORS = [
    [140, 90, 60],    # ground: brown
    [180, 180, 180],  # not_ground: grey
    [0, 0, 0],        # ignored: black
]

NORWAY_COMBINED_BINARY_NUM_CLASSES = 2
