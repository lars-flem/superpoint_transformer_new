"""Tile split and class mapping for a focused Bergen2022 3-class experiment.

The TILES dict is populated by scripts/national_model/setup_bergen2022_steep.py,
which selects the 30 steepest tiles in Bergen2022 (by elevation range) and
splits them 20/5/5 with a seeded shuffle.
"""

import numpy as np

from src.datasets.bergen2022_steep_tiles import TILES

__all__ = [
    "TILES",
    "ID2TRAINID",
    "CLASS_NAMES",
    "CLASS_COLORS",
    "BERGEN2022_STEEP_NUM_CLASSES",
]

# Bergen2022 standard LAS classes (per project report):
#   1 Uklassifisert, 2 Terreng, 3 Lav veg, 4 Med veg, 5 Høy veg,
#   6 Bygninger, 7 Støy, 9 Vann,
#   11 Vegbane, 13 Ledning jordlinje, 14 Ledning fase, 15 Mast, 17 Bro
#
# 3-class train IDs:
#   0 = ground, 1 = not_ground, 2 = building, 3 = ignored
ID2TRAINID = np.full(256, 3, dtype=np.int64)  # default: ignored

# Ground
ID2TRAINID[2] = 0    # Terreng
ID2TRAINID[11] = 0   # Vegbane

# Not ground (everything elevated except buildings)
ID2TRAINID[3] = 1    # Lav vegetasjon
ID2TRAINID[4] = 1    # Medium vegetasjon
ID2TRAINID[5] = 1    # Høy vegetasjon
ID2TRAINID[13] = 1   # Ledning beskyttelse (powerline ground line)
ID2TRAINID[14] = 1   # Ledning faselinje (powerline phase)
ID2TRAINID[15] = 1   # Mast
ID2TRAINID[17] = 1   # Bro

# Building
ID2TRAINID[6] = 2    # Bygninger

# Explicitly ignored
ID2TRAINID[1] = 3    # Uklassifisert
ID2TRAINID[7] = 3    # Støy
ID2TRAINID[9] = 3    # Vann

CLASS_NAMES = ["ground", "not_ground", "building", "ignored"]
CLASS_COLORS = [
    [140, 90, 60],    # ground: brown
    [180, 180, 180],  # not_ground: grey
    [255, 210, 0],    # building: yellow
    [0, 0, 0],        # ignored: black
]

BERGEN2022_STEEP_NUM_CLASSES = 3
