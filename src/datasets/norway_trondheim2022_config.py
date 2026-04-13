# src/datasets/norway_trondheim2022_config.py
import numpy as np


# TRONDHEIM 2022 30PKT - TRAIN/VAL/TEST SPLIT
# Total: 50 tiles - Train: 35, Val: 7, Test: 8
# Split chosen to balance:
# - number of tiles
# - total point count
# - raw LAS class distribution

TILES = {
    "train": [
        "32-1-510-215-43",
        "32-1-510-215-44",
        "32-1-510-215-52",
        "32-1-510-215-53",
        "32-1-510-215-54",
        "32-1-510-215-63",
        "32-1-510-215-64",
        "32-1-510-215-65",
        "32-1-510-215-70",
        "32-1-510-215-72",
        "32-1-510-215-73",
        "32-1-510-215-76",
        "32-1-511-215-00",
        "32-1-511-215-03",
        "32-1-511-215-05",
        "32-1-511-215-06",
        "32-1-511-215-07",
        "32-1-511-215-11",
        "32-1-511-215-12",
        "32-1-511-215-14",
        "32-1-511-215-17",
        "32-1-511-215-20",
        "32-1-511-215-21",
        "32-1-511-215-22",
        "32-1-511-215-23",
        "32-1-511-215-24",
        "32-1-511-215-25",
        "32-1-511-215-26",
        "32-1-511-215-33",
        "32-1-511-215-34",
        "32-1-511-215-35",
        "32-1-511-215-36",
        "32-1-511-215-37",
        "32-1-511-216-20",
        "32-1-511-216-30",
    ],
    "val": [
        "32-1-510-215-61",
        "32-1-510-215-62",
        "32-1-510-215-71",
        "32-1-511-215-01",
        "32-1-511-215-04",
        "32-1-511-215-13",
        "32-1-511-215-16",
    ],
    "test": [
        "32-1-510-215-60",
        "32-1-510-215-74",
        "32-1-510-215-75",
        "32-1-511-215-02",
        "32-1-511-215-10",
        "32-1-511-215-15",
        "32-1-511-215-27",
        "32-1-511-216-10",
    ],
}


# Mapping from LAS classification (0-255) -> train id
# 0 = ground
# 1 = not_ground
# 2 = ignored
ID2TRAINID = np.full(256, 2, dtype=np.int64)

# Ground
ID2TRAINID[2] = 0

# Vegetation / objects -> not_ground
ID2TRAINID[3] = 1    # Low vegetation
ID2TRAINID[4] = 1    # Medium vegetation
ID2TRAINID[5] = 1    # High vegetation
ID2TRAINID[14] = 1   # Powerlines
ID2TRAINID[15] = 1   # Transmission tower
ID2TRAINID[17] = 1   # Bridge

# Buildings -> not_ground
ID2TRAINID[6] = 1

# Ignored
ID2TRAINID[1] = 2    # Unclassified
ID2TRAINID[7] = 2    # Noise
ID2TRAINID[9] = 2    # Water


CLASS_NAMES = [
    "ground",
    "not_ground",
    "ignored",
]

CLASS_COLORS = [
    [140, 90, 60],    # ground
    [180, 180, 180],  # not_ground
    [0, 0, 0],        # ignored
]

NOR_NUM_CLASSES = 2
