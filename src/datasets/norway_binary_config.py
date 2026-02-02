# src/datasets/norway_binary_config.py
import numpy as np

# IDs (uten .laz) – må matche filnavnene dine i raw/{train,val,test}/
TILES = {
    "train": [
        "32-1-468-145-22",
        "32-1-468-145-23",
        # "32-1-468-145-24",  # tomme sub-tiles med xy_tiling=3
        # "32-1-468-145-25",  # tomme sub-tiles med xy_tiling=3
        "32-1-468-145-32",
        "32-1-468-145-33",
        "32-1-468-145-34",
        "32-1-468-145-35",
        "32-1-468-145-43",
        "32-1-468-145-44",
        "32-1-468-145-46",
        "32-1-468-145-47",
        "32-1-468-145-53",
        "32-1-468-145-56",
    ],
    "val": [
        "32-1-468-145-42",
        "32-1-468-145-54",
    ],
    "test": [
        "32-1-468-145-52",
        "32-1-468-145-55",
    ],
}

# mapping fra LAS classification (0-255) -> train id (0/1/2)
# 0 = ground, 1 = not_ground, 2 = void/ignored
ID2TRAINID = np.full(256, 2, dtype=np.int64)  # default: ignored

# LAS standard klassifisering:
# 1 = Unclassified      -> ignored (2)
# 2 = Ground            -> ground (0)
# 3 = Low Vegetation    -> not_ground (1)
# 4 = Medium Vegetation -> not_ground (1)
# 5 = High Vegetation   -> not_ground (1)
# 6 = Building          -> not_ground (1)
# 7 = Low Point (Noise) -> ignored (2)
# 9 = Water            -> ground (0)
# 14= Powerlines       -> not_ground (1)
# 15= Transmission Tower-> not_ground (1)
# 17= Bridge           -> not_ground (1)
# 24= Snow             -> ignored (2)

ID2TRAINID[2] = 0                  # Ground -> ground
ID2TRAINID[9] = 0                  # Water -> ground
ID2TRAINID[3] = 1                  # Low Vegetation -> not_ground
ID2TRAINID[4] = 1                  # Medium Vegetation -> not_ground
ID2TRAINID[5] = 1                  # High Vegetation -> not_ground
ID2TRAINID[6] = 1                  # Building -> not_ground
ID2TRAINID[14]= 1                  # Powerlines -> not_ground
ID2TRAINID[15]= 1                  # Transmission Tower -> not_ground
ID2TRAINID[17]= 1                  # Bridge -> not_ground
ID2TRAINID[24]= 2                  # Snow -> ignored

# Klasse 1 og 7 forblir 2 (ignored)

CLASS_NAMES = ["ground", "not_ground", "ignored"]
CLASS_COLORS = [
    [140, 90, 60],   # ground
    [180, 180, 180], # not_ground
    [0, 0, 0],       # ignored
]

NOR_NUM_CLASSES = 2
