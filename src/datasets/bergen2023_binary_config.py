# src/datasets/norway_binary_config.py
import numpy as np

# IDs (uten .laz) – må matche filnavnene dine i raw/{train,val,test}/
TILES = {
    "train": [
        "32-1-468-145-27",
        "32-1-468-145-32",
        "32-1-468-145-33",
        "32-1-468-145-34",
        "32-1-468-145-36",
        "32-1-468-145-37",
        "32-1-468-145-42",
        "32-1-468-145-43",
        "32-1-468-145-44",
        "32-1-468-145-45",
        "32-1-468-145-46",
        "32-1-468-145-47",
        "32-1-468-145-53",
        "32-1-468-145-54",
        "32-1-468-145-55",
        "32-1-468-145-56",
        "32-1-468-145-57",
        "32-1-468-145-62",
        "32-1-468-145-63",
        "32-1-468-145-64",
        "32-1-468-145-65",
        "32-1-468-145-66",
        "32-1-468-145-67",
        "32-1-468-145-72",
        "32-1-468-145-73",
        "32-1-468-145-74",
        "32-1-468-145-75",
        "32-1-468-145-76",
        "32-1-468-145-77",
        "32-1-468-146-30",
        "32-1-468-146-31",
        "32-1-468-146-32",
        "32-1-468-146-33",
        "32-1-468-146-34",
        "32-1-468-146-40",
        "32-1-468-146-41",
        "32-1-468-146-42",
        "32-1-468-146-43",
        "32-1-468-146-44",
        "32-1-468-146-45",
        "32-1-468-146-50",
        "32-1-468-146-51",
        "32-1-468-146-52",
        "32-1-468-146-53",
        "32-1-468-146-54",
        "32-1-468-146-55",
        "32-1-468-146-60",
        "32-1-468-146-61",
        "32-1-468-146-62",
        "32-1-468-146-63",
        "32-1-468-146-64",
        "32-1-468-146-65",
        "32-1-468-146-70",
        "32-1-468-146-71",
        "32-1-468-146-72",
        "32-1-468-146-73",
        "32-1-468-146-74",
        "32-1-468-146-75",
        "32-1-469-145-02",
        "32-1-469-145-03",
    ],
    "val": [
        "32-1-469-146-00",
        "32-1-469-146-01",
        "32-1-469-146-02",
        "32-1-469-146-03",
        "32-1-469-146-04",
        "32-1-469-146-05",
        "32-1-469-146-10",
        "32-1-469-146-11",
        "32-1-469-146-12",
        "32-1-469-146-13",
        "32-1-469-146-14",
        "32-1-469-146-15",
    ],
    "test": [
        "32-1-469-145-04",
        "32-1-469-145-05",
        "32-1-469-145-06",
        "32-1-469-145-07",
        "32-1-469-145-12",
        "32-1-469-145-13",
        "32-1-469-145-14",
        "32-1-469-145-15",
        "32-1-469-145-16",
        "32-1-469-145-17",
        "32-1-469-145-23",
        "32-1-469-145-24",
    ],
}

# mapping fra LAS classification (0-255) -> train id (0/1/2/3)
# 0 = terrain, 1 = building, 2 = bridge, 3 = ignored (not used in training)
ID2TRAINID = np.full(256, 3, dtype=np.int64)  # default: ignored (3)

# LAS standard klassifisering:
# 1 = Unclassified      -> ignored (3)
# 2 = Terrain            -> terrain (0)
# 6 = Building          -> building (1)
# 7 = Low Point (Noise) -> ignored (3)
# 9 = Water            -> ignored (3)
# 10 = Railway          -> ignored (3)
# 11 = Road             -> terrain (0)
# 13 = Groundlines      -> ignored (3)
# 14= Powerlines       -> ignored (3)
# 15= Transmission Tower-> ignored (3)
# 17= Bridge           -> bridge (2)
# 19 = wtf              -> ignored (3)
# 21= Snow             -> ignored (3)
# 22 = Temporal Exclusion -> ignored (3)

ID2TRAINID[1] = 3                  # Unclassified -> ignored
ID2TRAINID[2] = 0                  # Terrain -> terrain
ID2TRAINID[6] = 1                  # Building -> building
ID2TRAINID[7] = 3                  # Low Point (Noise) -> ignored
ID2TRAINID[9] = 3                  # Water -> ignored
ID2TRAINID[10] = 3                 # Railway -> ignored
ID2TRAINID[11] = 0                 # Road -> terrain
ID2TRAINID[13] = 3                 # Groundlines -> ignored
ID2TRAINID[14] = 3                 # Powerlines -> ignored
ID2TRAINID[15] = 3                 # Transmission Tower -> ignored
ID2TRAINID[17] = 2                 # Bridge -> bridge
ID2TRAINID[19] = 3                 # wtf -> ignored
ID2TRAINID[21] = 3                 # Snow -> ignored
ID2TRAINID[22] = 3                 # Temporal Exclusion -> ignored

CLASS_NAMES = ["terrain", "building", "bridge", "ignored"]
CLASS_COLORS = [
    [140, 90, 60],   # terrain (brown)
    [180, 180, 180], # building (gray)
    [100, 150, 200], # bridge (blue)
    [0, 0, 0],       # ignored (black)
]

NOR_NUM_CLASSES = 3  # terrain, building, bridge count for training (ignored class not counted)