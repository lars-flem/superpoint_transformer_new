"""Set up a focused, stratified Bergen2022 dataset spanning a range of
terrain steepness.

Sampling strategy (30 tiles total):
  - 15 steep tiles (z_range >= STEEP_MIN, sorted by z_range desc)
  - 10 medium tiles (MEDIUM_LO <= z_range <= MEDIUM_HI, random)
  - 5 flat tiles (z_range <= FLAT_MAX, random)

All tiles must have >= MIN_POINTS source points (filters out water-dominated
edge tiles where high steepness is an artifact).

Stratified split (20/5/5):
  steep  15 -> 10 train / 2 val / 3 test
  medium 10 ->  7 train / 2 val / 1 test
  flat    5 ->  3 train / 1 val / 1 test

FORCE_INCLUDE_TEST: tiles in this list are always placed in test, taking one
of the slots from their own stratum. A force-included medium tile reduces the
random medium pool from 10 to 9 etc.

Idempotent w.r.t. the seed: re-running produces the same selection. Symlinks
for tiles not in the new selection are removed; new ones are created.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import laspy

SRC_DIR = Path("/cluster/home/jakobep/bergen2022/data")
OUT_ROOT = Path("/cluster/home/jakobep/datasets/bergen2022_steep_3class")
REPO = Path("/cluster/home/jakobep/superpoint_transformer_new")
TILES_PY = REPO / "src" / "datasets" / "bergen2022_steep_tiles.py"

# Steepness band thresholds (m). Bergen is mountainous — there are essentially
# no near-zero z_range tiles with usable point density, so "flat" here means
# "the least-steep tiles Bergen actually offers" (30–80m, all hill-and-fjord).
# Per-band counts at >=5M points: steep 34, medium 66, flat 17.
STEEP_MIN = 250.0
MEDIUM_LO = 100.0
MEDIUM_HI = 200.0
FLAT_MIN = 30.0
FLAT_MAX = 80.0

N_STEEP = 15
N_MEDIUM = 10
N_FLAT = 5

# Per-stratum split (train, val, test) summing to count above
SPLIT_STEEP = (10, 2, 3)
SPLIT_MEDIUM = (7, 2, 1)
SPLIT_FLAT = (3, 1, 1)

MIN_POINTS = 5_000_000
SEED = 42

# Tiles forced into the test set (override stratified random selection).
# The tile's stratum is detected and one random medium/steep/flat test slot
# is replaced.
FORCE_INCLUDE_TEST = ["32-1-467-145-40"]

# Tiles to exclude from selection (flagged for label / intensity issues during
# manual inspection). When the TILES file already exists and contains any of
# these, the script runs in *surgical replacement* mode: it loads the existing
# selection, removes only the excluded tiles, and picks same-stratum
# replacements from the rest of the source pool. Tiles not in EXCLUDE are
# preserved exactly.
EXCLUDE = [
    # Medium tiles in train — flagged for misclassified trees / weird intensity
    "32-1-467-144-54",
    "32-1-467-145-62",
    "32-1-468-145-13",
    "32-1-468-145-23",
    # Flat tile in val — flagged for weird intensity
    "32-1-467-145-20",
]
REPLACEMENT_SEED = 1234  # different seed so replacements don't repeat earlier picks


def discover_and_score() -> list[tuple[float, int, str]]:
    """[(z_range, n_points, base_id)] sorted by steepness desc."""
    out = []
    for f in sorted(SRC_DIR.glob("*.laz")):
        h = laspy.open(f).header
        out.append((h.maxs[2] - h.mins[2], h.point_count, f.stem))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def classify(z_range: float) -> str | None:
    if z_range >= STEEP_MIN:
        return "steep"
    if MEDIUM_LO <= z_range <= MEDIUM_HI:
        return "medium"
    if FLAT_MIN <= z_range <= FLAT_MAX:
        return "flat"
    return None  # in a gap band — not sampled


def pick_stratum(
    scored: list[tuple[float, int, str]],
    stratum: str,
    n: int,
    rng: random.Random,
    exclude: set[str],
) -> list[tuple[float, int, str]]:
    pool = [
        s for s in scored
        if s[1] >= MIN_POINTS
        and classify(s[0]) == stratum
        and s[2] not in exclude
    ]
    if stratum == "steep":
        # take the steepest n
        return pool[:n]
    # for medium/flat, random sample within the band
    if len(pool) < n:
        raise ValueError(f"Only {len(pool)} {stratum} tiles available; need {n}")
    return rng.sample(pool, n)


def stratified_split(
    tiles: list[tuple[float, int, str]],
    split: tuple[int, int, int],
    rng: random.Random,
) -> dict[str, list[tuple[float, int, str]]]:
    n_train, n_val, n_test = split
    assert n_train + n_val + n_test == len(tiles), (
        f"{n_train}+{n_val}+{n_test} != {len(tiles)}"
    )
    shuffled = list(tiles)
    rng.shuffle(shuffled)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def make_symlinks(
    splits: dict[str, list[tuple[float, int, str]]],
) -> None:
    # Set of (split, name) we want to exist
    wanted: dict[str, set[str]] = {s: set() for s in ("train", "val", "test")}
    for split_name, tiles in splits.items():
        for _, _, base_id in tiles:
            wanted[split_name].add(f"{base_id}.laz")

    n_created = n_removed = n_kept = 0
    for split_name in ("train", "val", "test"):
        target_dir = OUT_ROOT / "raw" / split_name
        target_dir.mkdir(parents=True, exist_ok=True)
        # remove stale symlinks
        for existing in target_dir.iterdir():
            if existing.name not in wanted[split_name] and existing.is_symlink():
                existing.unlink()
                n_removed += 1
        # create missing
        for name in wanted[split_name]:
            link = target_dir / name
            if link.is_symlink() or link.exists():
                n_kept += 1
                continue
            src = SRC_DIR / name
            if not src.exists():
                raise FileNotFoundError(f"missing source: {src}")
            os.symlink(src, link)
            n_created += 1
    print(f"  Symlinks: created {n_created}, kept {n_kept}, removed {n_removed}")


def render_tiles_py(
    splits: dict[str, list[tuple[float, int, str]]],
    stratum_of: dict[str, str],
) -> str:
    lines = [
        '"""Auto-generated by scripts/national_model/setup_bergen2022_steep.py.',
        "",
        "Stratified Bergen2022 sample of 30 tiles spanning a range of",
        f"terrain steepness: {N_STEEP} steep (z_range >= {int(STEEP_MIN)}m),",
        f"{N_MEDIUM} medium ({int(MEDIUM_LO)}-{int(MEDIUM_HI)}m), {N_FLAT} flat",
        f"({int(FLAT_MIN)}-{int(FLAT_MAX)}m). Split 20/5/5, stratified, with",
        f"force-include test list: {FORCE_INCLUDE_TEST}.",
        '"""',
        "",
        "TILES = {",
    ]
    for split_name in ("train", "val", "test"):
        lines.append(f'    "{split_name}": [')
        # sort within each split by stratum then z_range desc
        tiles = sorted(
            splits[split_name],
            key=lambda t: ({"steep": 0, "medium": 1, "flat": 2}[stratum_of[t[2]]], -t[0]),
        )
        for z_range, n, tid in tiles:
            tag = stratum_of[tid]
            lines.append(f'        "{tid}",  # {tag:6s}  z_range={z_range:.1f}m  n={n:,}')
        lines.append("    ],")
    lines.append("}")
    return "\n".join(lines) + "\n"


def load_current_tiles() -> dict[str, list[str]] | None:
    """Load the existing TILES dict from the auto-generated tiles file.
    Returns None if the file doesn't exist or has no TILES."""
    if not TILES_PY.exists():
        return None
    ns: dict = {}
    exec(TILES_PY.read_text(), ns)
    return ns.get("TILES")


def surgical_replace(
    scored: list[tuple[float, int, str]],
    current_tiles: dict[str, list[str]],
    exclude_set: set[str],
) -> tuple[dict[str, list[tuple[float, int, str]]], dict[str, str], list[tuple[str, str, str]]]:
    """For each excluded tile in current_tiles, pick a same-stratum replacement
    not currently in use and not in EXCLUDE. Preserve every other tile.
    Returns (new_splits, stratum_of, replacement_log).
    """
    by_id = {s[2]: s for s in scored}
    in_use = {tid for tiles in current_tiles.values() for tid in tiles}
    rng = random.Random(REPLACEMENT_SEED)

    new_splits: dict[str, list[tuple[float, int, str]]] = {"train": [], "val": [], "test": []}
    stratum_of: dict[str, str] = {}
    log: list[tuple[str, str, str]] = []  # (split, old_tid, new_tid)

    for split_name, tile_ids in current_tiles.items():
        for tid in tile_ids:
            if tid not in exclude_set:
                # keep as-is
                z_range, n, _ = by_id[tid]
                new_splits[split_name].append(by_id[tid])
                stratum_of[tid] = classify(z_range)
                continue

            # Need replacement
            stratum = classify(by_id[tid][0])
            candidates = [
                s for s in scored
                if s[1] >= MIN_POINTS
                and classify(s[0]) == stratum
                and s[2] not in in_use
                and s[2] not in exclude_set
            ]
            if not candidates:
                raise ValueError(
                    f"No replacement available for {tid} (stratum={stratum})"
                )
            if stratum == "steep":
                # pick steepest available
                replacement = candidates[0]
            else:
                replacement = rng.choice(candidates)
            new_splits[split_name].append(replacement)
            stratum_of[replacement[2]] = stratum
            in_use.add(replacement[2])
            log.append((split_name, tid, replacement[2]))

    return new_splits, stratum_of, log


def main() -> None:
    print("Scoring tiles by elevation range...")
    scored = discover_and_score()
    print(f"  scanned {len(scored)} tiles")

    by_id = {s[2]: s for s in scored}

    # Detect surgical-replacement mode: TILES exists AND contains excluded tiles.
    current = load_current_tiles()
    exclude_set = set(EXCLUDE)
    needs_surgical = (
        current is not None
        and exclude_set
        and any(tid in exclude_set for tiles in current.values() for tid in tiles)
    )

    if needs_surgical:
        print(f"\nFound existing TILES with {len(exclude_set)} excluded entries.")
        print("Running in SURGICAL REPLACEMENT mode (preserving non-excluded tiles).")
        splits, stratum_of, log = surgical_replace(scored, current, exclude_set)
        print(f"\nReplacements:")
        for split_name, old, new in log:
            old_z = by_id[old][0]
            new_z = by_id[new][0]
            print(f"  {split_name}: {old} (z={old_z:.1f}m) → {new} (z={new_z:.1f}m)")
        for s in ("train", "val", "test"):
            print(f"  {s}: {len(splits[s])} tiles")

        print("\nCreating/updating symlinks...")
        make_symlinks(splits)

        print(f"\nWriting {TILES_PY.relative_to(REPO)}...")
        TILES_PY.write_text(render_tiles_py(splits, stratum_of))

        print("\nDone.")
        return

    # Fresh stratified pick (initial run, or no excluded tiles in current selection)
    rng = random.Random(SEED)

    # Force-included tiles: classify each and reserve a test slot for it.
    forced_by_stratum: dict[str, list[tuple[float, int, str]]] = {
        "steep": [], "medium": [], "flat": [],
    }
    for tid in FORCE_INCLUDE_TEST:
        if tid not in by_id:
            raise ValueError(f"force-include tile {tid!r} not in source")
        z_range, n, _ = by_id[tid]
        stratum = classify(z_range)
        if stratum is None:
            raise ValueError(
                f"force-include tile {tid} (z_range={z_range:.1f}m) doesn't "
                f"fall in any sampling band"
            )
        forced_by_stratum[stratum].append(by_id[tid])
        print(
            f"  forcing {tid} into TEST "
            f"(stratum={stratum}, z_range={z_range:.1f}m)"
        )

    # Pick stratum pools, excluding forced ones
    forced_ids = set(FORCE_INCLUDE_TEST)
    print("\nPicking strata...")
    steep = pick_stratum(scored, "steep", N_STEEP - len(forced_by_stratum["steep"]),
                         rng, forced_ids) + forced_by_stratum["steep"]
    medium = pick_stratum(scored, "medium", N_MEDIUM - len(forced_by_stratum["medium"]),
                          rng, forced_ids) + forced_by_stratum["medium"]
    flat = pick_stratum(scored, "flat", N_FLAT - len(forced_by_stratum["flat"]),
                        rng, forced_ids) + forced_by_stratum["flat"]
    print(f"  steep:  {len(steep)}  ({steep[-1][0]:.1f} – {steep[0][0]:.1f} m)")
    medium_sorted = sorted(medium, key=lambda x: x[0])
    print(f"  medium: {len(medium)} ({medium_sorted[0][0]:.1f} – {medium_sorted[-1][0]:.1f} m)")
    flat_sorted = sorted(flat, key=lambda x: x[0])
    print(f"  flat:   {len(flat)}  ({flat_sorted[0][0]:.1f} – {flat_sorted[-1][0]:.1f} m)")

    # Stratified split. Pull forced ones aside before shuffling, then put them
    # in test.
    print("\nSplitting (stratified, force-includes go to test)...")
    splits: dict[str, list[tuple[float, int, str]]] = {"train": [], "val": [], "test": []}
    for tiles, split_counts, name in (
        (steep, SPLIT_STEEP, "steep"),
        (medium, SPLIT_MEDIUM, "medium"),
        (flat, SPLIT_FLAT, "flat"),
    ):
        forced = [t for t in tiles if t[2] in forced_ids]
        unforced = [t for t in tiles if t[2] not in forced_ids]
        # Adjust test count by forced size
        n_train, n_val, n_test = split_counts
        n_test_random = n_test - len(forced)
        if n_test_random < 0:
            raise ValueError(
                f"Too many force-includes in {name} stratum "
                f"({len(forced)} > {n_test})"
            )
        rng.shuffle(unforced)
        splits["train"].extend(unforced[:n_train])
        splits["val"].extend(unforced[n_train : n_train + n_val])
        splits["test"].extend(unforced[n_train + n_val : n_train + n_val + n_test_random])
        splits["test"].extend(forced)

    for s in ("train", "val", "test"):
        print(f"  {s}: {len(splits[s])} tiles")

    # Stratum lookup for renderer
    stratum_of: dict[str, str] = {}
    for t in steep:  stratum_of[t[2]] = "steep"
    for t in medium: stratum_of[t[2]] = "medium"
    for t in flat:   stratum_of[t[2]] = "flat"

    print("\nCreating/updating symlinks...")
    make_symlinks(splits)

    print(f"\nWriting {TILES_PY.relative_to(REPO)}...")
    TILES_PY.write_text(render_tiles_py(splits, stratum_of))

    print("\nDone.")


if __name__ == "__main__":
    main()
