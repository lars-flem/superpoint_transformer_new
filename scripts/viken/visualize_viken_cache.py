"""Visualize ground-truth labels stored in the existing viken2022 cache.

Loads the preprocessed `.h5` files directly (no model, no reprocessing) so we
can inspect what label scheme the cache actually holds — useful for telling
whether a prior run was binary (2 classes) or multiclass (3+ classes), since
the histogram of label IDs in the cache will reveal it immediately.

Mirrors the structure of `visualize_trondheim_density.py` and
`visualize_gjerdrum_viken.py`: writes one full-tile HTML and one crop HTML
per sub-tile into `visualizations/viken_cache/`.
"""

import os
import sys
import glob
import torch

REPO = "/cluster/home/jakobep/superpoint_transformer_new"
sys.path.insert(0, REPO)

from src.data import NAG  # noqa: E402
from src.datasets.viken2022_config import (  # noqa: E402
    CLASS_COLORS,
    CLASS_NAMES,
    VIKEN2022_NUM_CLASSES,
)

OUT_DIR = os.path.join(REPO, "visualizations", "viken_cache")
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_DIR = "/cluster/home/jakobep/datasets/viken2022/processed"
CACHE_HASH = "999b969bc9e7f45e48797db8a658cff9"
TEST_TILE = "32-1-513-134-62"
CROP_RADIUS = 50

CLASS_NAMES_LIST = list(CLASS_NAMES)
CLASS_COLORS_LIST = list(CLASS_COLORS)
STUFF_CLASSES = list(range(VIKEN2022_NUM_CLASSES))


def find_subtile_files() -> list:
    """Return all cached sub-tile files matching TEST_TILE across splits."""
    matches = []
    for split in ("test", "val", "train"):
        pattern = os.path.join(
            CACHE_DIR, split, CACHE_HASH, f"{TEST_TILE}*.h5"
        )
        for p in sorted(glob.glob(pattern)):
            matches.append((split, p))
    return matches


def visualize_subtile(split: str, h5_path: str) -> None:
    name = os.path.splitext(os.path.basename(h5_path))[0]
    print(f"\n=== {split}/{name} ===")
    print(f"  path: {h5_path}")

    nag = NAG.load(h5_path)
    print(f"  levels: {nag.num_levels}  level-0 nodes: {nag[0].pos.shape[0]:,}")

    # Ground-truth label histogram. y is stored as a per-superpoint class
    # histogram (CSR). Per-point label = argmax over classes.
    y = nag[0].y
    if y is None:
        print("  (no y in cache — nothing to visualize)")
        return

    if y.dim() == 2:
        n_classes_in_cache = y.shape[1]
        per_node_label = y.argmax(dim=1)
        counts = y.sum(dim=0).long().tolist()
        print(f"  cache n_classes (y.shape[1]): {n_classes_in_cache}")
        print(f"  per-class point counts: {counts}")
    else:
        n_classes_in_cache = int(y.max().item()) + 1
        per_node_label = y
        uniq, cnts = torch.unique(y, return_counts=True)
        print(f"  cache class IDs (unique y): "
              f"{dict(zip(uniq.tolist(), cnts.tolist()))}")

    # Pad class names/colors if cache holds more classes than current scheme.
    n_show = max(n_classes_in_cache, VIKEN2022_NUM_CLASSES)
    names = list(CLASS_NAMES_LIST)
    colors = list(CLASS_COLORS_LIST)
    while len(names) < n_show:
        names.append(f"class_{len(names)}")
        colors.append([255, 0, 255])  # magenta sentinel for unexpected ids

    common_kwargs = dict(
        class_names=names,
        class_colors=colors,
        stuff_classes=list(range(n_show)),
        num_classes=n_show,
        max_points=100_000,
    )

    full_path = os.path.join(OUT_DIR, f"{name}_full.html")
    nag.show(
        figsize=1600,
        title=f"viken cache GT — {name} (split={split})",
        path=full_path,
        display=False,
        **common_kwargs,
    )
    print(f"  wrote: {full_path}")

    center = nag[0].pos.mean(dim=0).view(1, -1)
    crop_path = os.path.join(OUT_DIR, f"{name}_crop.html")
    nag.show(
        figsize=1600,
        radius=CROP_RADIUS,
        center=center,
        title=f"viken cache GT — {name} {CROP_RADIUS} m crop",
        path=crop_path,
        display=False,
        **common_kwargs,
    )
    print(f"  wrote: {crop_path}")


def main() -> None:
    files = find_subtile_files()
    if not files:
        print(f"No cached sub-tiles found for {TEST_TILE} under {CACHE_DIR}")
        return
    print(f"Found {len(files)} cached sub-tile(s) for {TEST_TILE}.")
    for split, path in files:
        try:
            visualize_subtile(split, path)
        except Exception as exc:
            print(f"!! failed on {path}: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
