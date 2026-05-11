import numpy as np

# IDs must match file names in raw/{train,val,test}/ without the .laz suffix.
#
# Spatial split strategy (49 tiles total):
#   train (35): 517-137 rows 4-7 (29) + 518-137 rows 00-05 (6)
#   val   ( 6): 517-137 row 3 (4) + 518-136 block (2)
#   test  ( 8): 518-137 rows 10-31 (8)

TILES = {
    "train": [
        # 517-137 row 4 (7 tiles)
        "32-1-517-137-41", "32-1-517-137-42", "32-1-517-137-43",
        "32-1-517-137-44", "32-1-517-137-45", "32-1-517-137-46", "32-1-517-137-47",
        # 517-137 row 5 (8 tiles)
        "32-1-517-137-50", "32-1-517-137-51", "32-1-517-137-52", "32-1-517-137-53",
        "32-1-517-137-54", "32-1-517-137-55", "32-1-517-137-56", "32-1-517-137-57",
        # 517-137 row 6 (7 tiles)
        "32-1-517-137-60", "32-1-517-137-61", "32-1-517-137-62", "32-1-517-137-63",
        "32-1-517-137-64", "32-1-517-137-65", "32-1-517-137-66",
        # 517-137 row 7 (7 tiles)
        "32-1-517-137-70", "32-1-517-137-71", "32-1-517-137-72", "32-1-517-137-73",
        "32-1-517-137-74", "32-1-517-137-75", "32-1-517-137-76",
        # 518-137 rows 00-05 (6 tiles)
        "32-1-518-137-00", "32-1-518-137-01", "32-1-518-137-02",
        "32-1-518-137-03", "32-1-518-137-04", "32-1-518-137-05",
    ],
    "val": [
        # 517-137 row 3 (4 tiles)
        "32-1-517-137-33", "32-1-517-137-34", "32-1-517-137-35", "32-1-517-137-36",
        # 518-136 block (2 tiles)
        "32-1-518-136-17", "32-1-518-136-27",
    ],
    "test": [
        # 518-137 rows 10-31 (8 tiles)
        "32-1-518-137-10", "32-1-518-137-11", "32-1-518-137-12", "32-1-518-137-13",
        "32-1-518-137-20", "32-1-518-137-21", "32-1-518-137-22",
        "32-1-518-137-31",
    ],
}

# Mapping from LAS classification (0-255) -> train id.
# 0 = ground, 1 = not_ground.
#
# Classes present in Gjerdrum data:
#   1  → Uklassifisert (Unclassified)
#   2  → Terreng (Ground)
#   7  → Støy (Noise)
#   17 → Bro (Bridge) — included for safety, may not appear in practice
ID2TRAINID = np.full(256, 1, dtype=np.int64)  # default: not_ground

# Ground
ID2TRAINID[2] = 0   # Terreng

# not_ground (explicit, already covered by default)
ID2TRAINID[1] = 1   # Uklassifisert
ID2TRAINID[7] = 1   # Støy
ID2TRAINID[17] = 1  # Bro

CLASS_NAMES = ["ground", "not_ground"]
CLASS_COLORS = [
    [140, 90, 60],    # ground: brown
    [180, 180, 180],  # not_ground: grey
]

GJERDRUM_NUM_CLASSES = 2
