import numpy as np

# Identical split to bergen2020_config — kept here to avoid src import at module load time
TILES = {
    "train": [
        "32-1-468-145-23", "32-1-468-145-24", "32-1-468-145-25", "32-1-468-145-26",
        "32-1-468-145-32", "32-1-468-145-35", "32-1-468-145-36", "32-1-468-145-37",
        "32-1-468-145-43", "32-1-468-145-44", "32-1-468-145-45", "32-1-468-145-47",
        "32-1-468-145-53", "32-1-468-145-56", "32-1-468-145-57",
        "32-1-468-145-62", "32-1-468-145-63", "32-1-468-145-65", "32-1-468-145-66",
    ],
    "val":  ["32-1-468-145-34", "32-1-468-145-42", "32-1-468-145-54", "32-1-468-145-64"],
    "test": ["32-1-468-145-33", "32-1-468-145-46", "32-1-468-145-52", "32-1-468-145-55"],
}

# Binary: ground (0) vs not_ground (1)
# Ignored = 2

ID2TRAINID = np.full(256, 2, dtype=np.int64)

ID2TRAINID[2] = 0   # Ground
ID2TRAINID[3] = 1   # Low vegetation -> not_ground
ID2TRAINID[4] = 1   # Medium vegetation -> not_ground
ID2TRAINID[5] = 1   # High vegetation -> not_ground
ID2TRAINID[6] = 1   # Building -> not_ground
ID2TRAINID[14] = 1  # Powerlines -> not_ground
ID2TRAINID[15] = 1  # Transmission tower -> not_ground
ID2TRAINID[17] = 1  # Bridge -> not_ground
# 1 (unclassified), 7 (noise), 9 (water) -> ignored (default 2)

BERGEN2020_BINARY_NUM_CLASSES = 2

CLASS_NAMES = ["ground", "not_ground", "ignored"]
CLASS_COLORS = [
    [140, 90, 60],    # ground (brown)
    [180, 180, 180],  # not_ground (grey)
    [0, 0, 0],        # ignored
]
