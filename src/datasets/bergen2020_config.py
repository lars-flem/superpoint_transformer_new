import numpy as np

# Bergen 2020 5pkt — 28 aktive tiles (ekskludert: 22 liten/skog, 67 vann, 72/73/74 høy uklassifisert)
# Split: 23 train / 2 val / 2 test
TILES = {
    "train": [
        "32-1-468-145-23", "32-1-468-145-24", "32-1-468-145-25", "32-1-468-145-26",
        "32-1-468-145-32", "32-1-468-145-35", "32-1-468-145-36", "32-1-468-145-37",
        "32-1-468-145-43", "32-1-468-145-44", "32-1-468-145-45", "32-1-468-145-47",
        "32-1-468-145-53", "32-1-468-145-56", "32-1-468-145-57",
        "32-1-468-145-62", "32-1-468-145-63", "32-1-468-145-65", "32-1-468-145-66",
    ],
    "val": [
        "32-1-468-145-34",  # rad 3, 21.6% bygg
        "32-1-468-145-42",  # rad 4, 20.6% bygg (fra norway_binary)
        "32-1-468-145-54",  # rad 5, 10.0% bygg (fra norway_binary)
        "32-1-468-145-64",  # rad 6, 0% bygg (negativ eksempel)
    ],
    "test": [
        "32-1-468-145-33",  # rad 3,  9.8% bygg
        "32-1-468-145-46",  # rad 4, 14.3% bygg
        "32-1-468-145-52",  # rad 5, 18.7% bygg (fra norway_binary)
        "32-1-468-145-55",  # rad 5,  8.8% bygg (fra norway_binary)
    ],
}

# 3-class: ground / not_ground / building
# Ignored = 3
ID2TRAINID = np.full(256, 3, dtype=np.int64)

ID2TRAINID[2] = 0   # Ground
ID2TRAINID[3] = 1   # Low vegetation
ID2TRAINID[4] = 1   # Medium vegetation
ID2TRAINID[5] = 1   # High vegetation
ID2TRAINID[6] = 2   # Building
ID2TRAINID[14] = 1  # Powerlines
ID2TRAINID[15] = 1  # Transmission tower
ID2TRAINID[17] = 1  # Bridge
# 1 (unclassified), 7 (noise), 9 (water) -> ignored (default 3)

BERGEN2020_NUM_CLASSES = 3

CLASS_NAMES = ["ground", "not_ground", "building", "ignored"]
CLASS_COLORS = [
    [140, 90, 60],    # ground (brown)
    [120, 180, 80],   # not_ground (green)
    [220, 20, 60],    # building (red)
    [0, 0, 0],        # ignored
]
