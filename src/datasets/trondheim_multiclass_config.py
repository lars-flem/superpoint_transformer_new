import numpy as np

# IDs must match file names in raw/{train,val,test}/ without the .laz suffix.
# Same spatial train/val/test split as the binary trondheim experiment.

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
# Classes present in Trondheim data:
#   1  → Unclassified (58%) — ignored
#   2  → Ground (18%)       → class 0
#   3  → Low vegetation (4%) → class 1
#   4  → Medium vegetation (5%) → class 2
#   5  → High vegetation (15%) → class 3
#   7  → Noise (0.05%)      — ignored
#   17 → Bridge (0.2%)      → class 4
ID2TRAINID = np.full(256, 5, dtype=np.int64)  # default: ignored

ID2TRAINID[2] = 0   # Ground
ID2TRAINID[3] = 1   # Low vegetation
ID2TRAINID[4] = 2   # Medium vegetation
ID2TRAINID[5] = 3   # High vegetation
ID2TRAINID[17] = 4  # Bridge

# Explicitly ignored
ID2TRAINID[1] = 5   # Unclassified
ID2TRAINID[7] = 5   # Low point (noise)

CLASS_NAMES = ["ground", "low_veg", "med_veg", "high_veg", "bridge", "ignored"]
CLASS_COLORS = [
    [140, 90, 60],    # ground: brown
    [144, 238, 144],  # low_veg: light green
    [34, 139, 34],    # med_veg: medium green
    [0, 80, 0],       # high_veg: dark green
    [255, 128, 0],    # bridge: orange
    [0, 0, 0],        # ignored: black
]

TRONDHEIM_MULTICLASS_NUM_CLASSES = 5
