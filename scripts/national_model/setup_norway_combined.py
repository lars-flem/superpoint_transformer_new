"""
Set up the combined Norway dataset (Viken + Oslo + Bergen2022 + Bergen2020).

What this script does:
  1. Discovers source LAZ files in each of the 4 regions
  2. Performs a deterministic seeded 70/15/15 train/val/test split per region
  3. Creates symlinks under both
       /cluster/home/jakobep/datasets/norway_combined_3class/raw/{train,val,test}/
       /cluster/home/jakobep/datasets/norway_combined_binary/raw/{train,val,test}/
     using region-prefixed names (e.g. viken_<id>.laz, oslo_<id>.laz) to avoid
     collisions between bergen2020/bergen2022 (which share UTM tile ids).
  4. Generates src/datasets/norway_combined_tiles.py with TILES (full split) and
     TILES_MINI (5/2/2 per region, for development).

Idempotent: existing symlinks are skipped. Re-running with the same seed produces
the same split.

Usage:
    python scripts/setup_norway_combined.py
"""

from __future__ import annotations

import os
import os.path as osp
import random
from pathlib import Path

# Source data per region: (region_name, source_directory_glob_root)
# For Viken we point to the existing dataset/raw/{train,val,test} layout — we
# flatten across the existing split since the user wants a fresh split.
SOURCES = {
    "viken": [
        Path("/cluster/home/jakobep/datasets/viken2022/raw/train"),
        Path("/cluster/home/jakobep/datasets/viken2022/raw/val"),
        Path("/cluster/home/jakobep/datasets/viken2022/raw/test"),
    ],
    "oslo": [Path("/cluster/home/jakobep/6346/data")],
    "bergen2022": [Path("/cluster/home/jakobep/bergen2022/data")],
    "bergen2020": [Path("/cluster/home/jakobep/bergen2020/data")],
}

# Output dataset roots — both binary and 3class variants share the same raw
# tile inventory but live under separate roots so that the SPT preprocessing
# `processed/` folders don't collide.
OUT_ROOTS = [
    Path("/cluster/home/jakobep/datasets/norway_combined_3class"),
    Path("/cluster/home/jakobep/datasets/norway_combined_binary"),
]

# Tiles file in the SPT repo
REPO = Path("/cluster/home/jakobep/superpoint_transformer_new")
TILES_PY = REPO / "src" / "datasets" / "norway_combined_tiles.py"

# Absolute number of tiles per region (balanced sampling so each region
# contributes the same amount of data regardless of source size).
TILES_PER_REGION = {"train": 70, "val": 15, "test": 15}
MINI_PER_REGION = {"train": 5, "val": 2, "test": 2}
SEED = 42


def discover(region: str) -> list[str]:
    """Return sorted unique base ids (no extension) for a region."""
    seen: set[str] = set()
    for d in SOURCES[region]:
        if not d.exists():
            print(f"  [warn] {region}: missing source dir {d}")
            continue
        for laz in d.glob("*.laz"):
            seen.add(laz.stem)
    return sorted(seen)


def split_region(base_ids: list[str]) -> dict[str, list[str]]:
    """Deterministic seeded split, taking fixed absolute counts per region
    (TILES_PER_REGION). Tiles beyond train+val+test are dropped so each region
    contributes the same amount of data."""
    rng = random.Random(SEED)
    ids = list(base_ids)
    rng.shuffle(ids)
    n_train = TILES_PER_REGION["train"]
    n_val = TILES_PER_REGION["val"]
    n_test = TILES_PER_REGION["test"]
    needed = n_train + n_val + n_test
    if len(ids) < needed:
        raise ValueError(
            f"Region has only {len(ids)} tiles but {needed} requested "
            f"({n_train} train + {n_val} val + {n_test} test)"
        )
    return {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train : n_train + n_val]),
        "test": sorted(ids[n_train + n_val : n_train + n_val + n_test]),
    }


def source_path_for(region: str, base_id: str) -> Path:
    """Locate the actual LAZ file for (region, base_id) in the source dirs."""
    for d in SOURCES[region]:
        candidate = d / f"{base_id}.laz"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{region}/{base_id}.laz not found in any source")


def make_symlinks(per_region_split: dict[str, dict[str, list[str]]]) -> None:
    """Create symlinks. Wipes existing symlinks first to avoid stale entries
    when the selection changes (e.g., when TILES_PER_REGION is reduced)."""
    # Build the set of (out_root, split, link_name) we WANT to exist
    wanted: dict[tuple[Path, str], set[str]] = {}
    for out_root in OUT_ROOTS:
        for split in ("train", "val", "test"):
            wanted[(out_root, split)] = set()
    for region, splits in per_region_split.items():
        for split, base_ids in splits.items():
            for base_id in base_ids:
                name = f"{region}_{base_id}.laz"
                for out_root in OUT_ROOTS:
                    wanted[(out_root, split)].add(name)

    for out_root in OUT_ROOTS:
        n_removed = 0
        n_created = 0
        n_kept = 0
        for split in ("train", "val", "test"):
            target_dir = out_root / "raw" / split
            target_dir.mkdir(parents=True, exist_ok=True)
            # Remove any existing symlinks not in the wanted set
            wanted_names = wanted[(out_root, split)]
            for existing in target_dir.iterdir():
                if existing.name not in wanted_names and existing.is_symlink():
                    existing.unlink()
                    n_removed += 1
            # Create missing symlinks
            for region, splits in per_region_split.items():
                for s, base_ids in splits.items():
                    if s != split:
                        continue
                    for base_id in base_ids:
                        link = target_dir / f"{region}_{base_id}.laz"
                        if link.is_symlink() or link.exists():
                            n_kept += 1
                            continue
                        src = source_path_for(region, base_id)
                        os.symlink(src, link)
                        n_created += 1
        print(
            f"  {out_root.name}: created {n_created}, kept {n_kept}, "
            f"removed {n_removed}"
        )


def render_tiles_py(per_region_split: dict[str, dict[str, list[str]]]) -> str:
    """Build the norway_combined_tiles.py source string."""
    # Combine into the TILES dict using region-prefixed ids (matches symlink names)
    tiles: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for region in ("viken", "oslo", "bergen2022", "bergen2020"):
        for split in ("train", "val", "test"):
            tiles[split].extend(
                f"{region}_{b}" for b in per_region_split[region][split]
            )
    for split in tiles:
        tiles[split].sort()

    # Mini split: first N per region per split
    mini: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for region in ("viken", "oslo", "bergen2022", "bergen2020"):
        for split in ("train", "val", "test"):
            n = MINI_PER_REGION[split]
            picked = per_region_split[region][split][:n]
            mini[split].extend(f"{region}_{b}" for b in picked)
    for split in mini:
        mini[split].sort()

    def fmt(name: str, d: dict[str, list[str]]) -> str:
        out = [f"{name} = {{"]
        for split in ("train", "val", "test"):
            out.append(f'    "{split}": [')
            for tid in d[split]:
                out.append(f'        "{tid}",')
            out.append("    ],")
        out.append("}")
        return "\n".join(out)

    counts = {s: len(tiles[s]) for s in tiles}
    counts_mini = {s: len(mini[s]) for s in mini}
    region_counts = {
        r: {s: len(per_region_split[r][s]) for s in ("train", "val", "test")}
        for r in per_region_split
    }
    region_total = {r: sum(region_counts[r].values()) for r in region_counts}

    header = f'''"""
Auto-generated by scripts/setup_norway_combined.py — do not edit by hand.

Combined Norway dataset: Viken + Oslo + Bergen2022 + Bergen2020.
Tile ids are region-prefixed (e.g. "viken_32-1-510-131-05") to avoid collisions
across regions. Symlinks with these names live under
  /cluster/home/jakobep/datasets/norway_combined_{{3class,binary}}/raw/{{split}}/

Per-region tile counts (train / val / test):
  viken      : {region_counts["viken"]["train"]:>4} / {region_counts["viken"]["val"]:>4} / {region_counts["viken"]["test"]:>4}  (total {region_total["viken"]})
  oslo       : {region_counts["oslo"]["train"]:>4} / {region_counts["oslo"]["val"]:>4} / {region_counts["oslo"]["test"]:>4}  (total {region_total["oslo"]})
  bergen2022 : {region_counts["bergen2022"]["train"]:>4} / {region_counts["bergen2022"]["val"]:>4} / {region_counts["bergen2022"]["test"]:>4}  (total {region_total["bergen2022"]})
  bergen2020 : {region_counts["bergen2020"]["train"]:>4} / {region_counts["bergen2020"]["val"]:>4} / {region_counts["bergen2020"]["test"]:>4}  (total {region_total["bergen2020"]})
  COMBINED   : {counts["train"]:>4} / {counts["val"]:>4} / {counts["test"]:>4}  (total {sum(counts.values())})

Mini split (5/2/2 per region) for development:
  COMBINED   : {counts_mini["train"]:>4} / {counts_mini["val"]:>4} / {counts_mini["test"]:>4}  (total {sum(counts_mini.values())})

The previous Viken-only split (used in the viken2022 experiments) is kept as a
reference comment at the bottom of this file.
"""

REGIONS = ["viken", "oslo", "bergen2022", "bergen2020"]


'''

    legacy_viken = '''
# ----------------------------------------------------------------------------
# Reference: previous Viken-only split (47 train, 15 val, 15 test) — used in
# the viken2022 experiments. Preserved here in case it is needed later.
# ----------------------------------------------------------------------------
# VIKEN_LEGACY_TILES = {
#     "train": [
#         "32-1-510-131-05", "32-1-510-131-06", "32-1-510-131-15",
#         "32-1-510-131-16", "32-1-510-131-25", "32-1-510-131-26",
#         "32-1-510-131-30", "32-1-510-134-72", "32-1-510-134-73",
#         "32-1-511-131-00", "32-1-511-131-01", "32-1-511-131-02",
#         "32-1-511-131-03", "32-1-511-131-04", "32-1-511-132-51",
#         "32-1-511-132-52", "32-1-511-132-53", "32-1-511-132-54",
#         "32-1-511-132-55", "32-1-511-132-56", "32-1-511-132-57",
#         "32-1-511-134-27", "32-1-511-134-30", "32-1-511-134-31",
#         "32-1-511-134-32", "32-1-511-134-33", "32-1-511-134-34",
#         "32-1-511-134-45", "32-1-512-131-70", "32-1-512-131-71",
#         "32-1-512-131-72", "32-1-512-131-73", "32-1-512-131-74",
#         "32-1-512-131-75", "32-1-512-131-76", "32-1-512-133-33",
#         "32-1-512-133-34", "32-1-512-133-35", "32-1-512-133-36",
#         "32-1-512-133-37", "32-1-512-133-40", "32-1-512-134-76",
#         "32-1-512-134-77", "32-1-512-135-00", "32-1-512-135-01",
#         "32-1-512-135-02", "32-1-512-135-03",
#     ],
#     "val": [
#         "32-1-513-131-14", "32-1-513-131-16", "32-1-513-131-17",
#         "32-1-513-132-21", "32-1-513-132-22", "32-1-513-132-55",
#         "32-1-513-132-56", "32-1-513-133-11", "32-1-513-133-12",
#         "32-1-513-133-36", "32-1-513-133-37", "32-1-513-133-63",
#         "32-1-513-133-64", "32-1-513-134-10", "32-1-513-134-11",
#     ],
#     "test": [
#         "32-1-513-134-35", "32-1-513-134-36", "32-1-513-134-37",
#         "32-1-513-134-62", "32-1-513-134-63", "32-1-513-135-07",
#         "32-1-513-135-10", "32-1-513-135-34", "32-1-513-135-35",
#         "32-1-513-135-63", "32-1-513-135-64", "32-1-514-133-12",
#         "32-1-514-133-13", "32-1-514-134-07", "32-1-514-134-10",
#     ],
# }
'''

    return header + fmt("TILES", tiles) + "\n\n" + fmt("TILES_MINI", mini) + "\n" + legacy_viken


def main() -> None:
    print("Discovering source LAZ files...")
    per_region_ids = {r: discover(r) for r in SOURCES}
    for r, ids in per_region_ids.items():
        print(f"  {r:>10}: {len(ids):>4} tiles")
    total_unique = sum(len(v) for v in per_region_ids.values())
    print(f"  {'TOTAL':>10}: {total_unique:>4} (region-prefixed → no collisions)")

    print("\nSplitting per region (seeded 70/15/15)...")
    per_region_split = {r: split_region(ids) for r, ids in per_region_ids.items()}
    for r in per_region_ids:
        s = per_region_split[r]
        print(
            f"  {r:>10}: {len(s['train']):>4} / {len(s['val']):>4} / {len(s['test']):>4}"
        )

    print("\nCreating symlinks...")
    make_symlinks(per_region_split)

    print(f"\nWriting {TILES_PY.relative_to(REPO)}...")
    TILES_PY.write_text(render_tiles_py(per_region_split))

    print("\nDone.")


if __name__ == "__main__":
    main()
