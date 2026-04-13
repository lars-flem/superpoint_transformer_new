import numpy as np

# IDs must match file names in raw/{train,val,test}/ without the .laz suffix.
#
# Spatial split strategy (50 tiles total):
#   train (35): 510-215 block (all 18) + 511-215 rows 00-07 (8) + rows 20-23 (4) + rows 33-37 (5)
#   val   ( 8): 511-215 rows 10-17
#   test  ( 7): 511-215 rows 24-27 (4) + 511-216 block (3)

TILES = {
    "train": [
        # 510-215 block (18 tiles)
        "32-1-510-215-43", "32-1-510-215-44",
        "32-1-510-215-52", "32-1-510-215-53", "32-1-510-215-54",
        "32-1-510-215-60", "32-1-510-215-61", "32-1-510-215-62",
        "32-1-510-215-63", "32-1-510-215-64", "32-1-510-215-65",
        "32-1-510-215-70", "32-1-510-215-71", "32-1-510-215-72",
        "32-1-510-215-73", "32-1-510-215-74", "32-1-510-215-75",
        "32-1-510-215-76",
        # 511-215 rows 00-07 (8 tiles)
        "32-1-511-215-00", "32-1-511-215-01", "32-1-511-215-02",
        "32-1-511-215-03", "32-1-511-215-04", "32-1-511-215-05",
        "32-1-511-215-06", "32-1-511-215-07",
        # 511-215 rows 20-23 (4 tiles)
        "32-1-511-215-20", "32-1-511-215-21", "32-1-511-215-22",
        "32-1-511-215-23",
        # 511-215 rows 33-37 (5 tiles)
        "32-1-511-215-33", "32-1-511-215-34", "32-1-511-215-35",
        "32-1-511-215-36", "32-1-511-215-37",
    ],
    "val": [
        # 511-215 rows 10-17 (8 tiles)
        "32-1-511-215-10", "32-1-511-215-11", "32-1-511-215-12",
        "32-1-511-215-13", "32-1-511-215-14", "32-1-511-215-15",
        "32-1-511-215-16", "32-1-511-215-17",
    ],
    "test": [
        # 511-215 rows 24-27 (4 tiles)
        "32-1-511-215-24", "32-1-511-215-25", "32-1-511-215-26",
        "32-1-511-215-27",
        # 511-216 block (3 tiles)
        "32-1-511-216-10", "32-1-511-216-20", "32-1-511-216-30",
    ],
}

# Mapping from LAS classification (0-255) -> train id.
# 0 = ground, 1 = not_ground, 2 = ignored.
#
# Classes present in Trondheim 30pkt data:
#   1  → Unclassified (58%) — unknown content, kept as ignored
#   2  → Ground (18%)
#   3  → Low vegetation (4%)
#   4  → Medium vegetation (5%)
#   5  → High vegetation (15%)
#   7  → Low point / noise (0.05%)
#   17 → Bridge (0.2%)
#
# Classes 6 (building), 9 (water), 14-15 (powerlines/towers), 23 (grass)
# are NOT present in this dataset.
ID2TRAINID = np.full(256, 2, dtype=np.int64)  # default: ignored

# Ground
ID2TRAINID[2] = 0   # Ground

# Not ground
ID2TRAINID[3] = 1   # Low vegetation
ID2TRAINID[4] = 1   # Medium vegetation
ID2TRAINID[5] = 1   # High vegetation
ID2TRAINID[17] = 1  # Bridge

# Explicitly ignored (already covered by default, listed for clarity)
ID2TRAINID[1] = 2   # Unclassified — content unknown, exclude from training
ID2TRAINID[7] = 2   # Low point (noise)

CLASS_NAMES = ["ground", "not_ground", "ignored"]
CLASS_COLORS = [
    [140, 90, 60],    # ground: brown
    [180, 180, 180],  # not_ground: grey
    [0, 0, 0],        # ignored: black
]

TRONDHEIM_NUM_CLASSES = 2
