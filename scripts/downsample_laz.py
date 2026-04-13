"""
Downsample LAZ files to a target point density (points per square meter).

Strategy: 2D spatial grid (cell_size x cell_size meters). For each cell,
randomly keep at most `density` points. All point attributes are preserved.
Points in sparse cells (fewer than `density` points) are kept as-is.

Usage:
    python scripts/downsample_laz.py \\
        --input_dir /path/to/30pkt/raw \\
        --output_dir /path/to/20pkt/raw \\
        --density 20 \\
        --cell_size 1.0 \\
        --seed 42

    # Or process all splits at once by pointing to the parent:
    python scripts/downsample_laz.py \\
        --input_dir /path/to/trondheim30/raw \\
        --output_dir /path/to/trondheim20/raw \\
        --density 20
"""

import argparse
import os
import os.path as osp
from pathlib import Path

import laspy
import numpy as np


def downsample_laz(
    input_path: str,
    output_path: str,
    density: float,
    cell_size: float = 1.0,
    seed: int = 42,
) -> None:
    """
    Downsample a single LAZ file to target density using a 2D spatial grid.

    Args:
        input_path: Path to input .laz file.
        output_path: Path to write downsampled .laz file.
        density: Target points per square meter.
        cell_size: Grid cell size in meters (default 1.0 = 1m² cells).
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    las = laspy.read(input_path)
    n_in = len(las.x)

    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)

    # Compute 2D grid cell indices for each point
    x_min, y_min = x.min(), y.min()
    cell_x = np.floor((x - x_min) / cell_size).astype(np.int64)
    cell_y = np.floor((y - y_min) / cell_size).astype(np.int64)

    # Unique cell ID as a single integer
    x_cells = cell_x.max() + 1
    cell_id = cell_y * x_cells + cell_x

    # Sort points by cell for efficient grouping
    sort_idx = np.argsort(cell_id, kind="stable")
    cell_id_sorted = cell_id[sort_idx]

    # Find boundaries between cells
    boundaries = np.where(np.diff(cell_id_sorted))[0] + 1
    boundaries = np.concatenate([[0], boundaries, [n_in]])

    # For each cell: randomly keep at most `density * cell_size^2` points
    max_per_cell = max(1, int(round(density * cell_size * cell_size)))
    keep_indices = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        n_cell = end - start
        if n_cell <= max_per_cell:
            keep_indices.append(sort_idx[start:end])
        else:
            chosen = rng.choice(n_cell, size=max_per_cell, replace=False)
            keep_indices.append(sort_idx[start + chosen])

    keep = np.concatenate(keep_indices)
    keep.sort()  # Restore original order for cleaner output

    n_out = len(keep)
    area = (x.max() - x_min) * (y.max() - y_min)
    actual_density = n_out / area if area > 0 else 0

    print(
        f"  {osp.basename(input_path)}: "
        f"{n_in:,} → {n_out:,} pts  "
        f"({actual_density:.1f} pkt/m²)"
    )

    # Write output LAZ preserving all point format dimensions
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    out_las = laspy.LasData(header=las.header)
    out_las.points = las.points[keep]
    out_las.write(output_path)


def process_directory(
    input_dir: str,
    output_dir: str,
    density: float,
    cell_size: float = 1.0,
    seed: int = 42,
) -> None:
    """
    Process all .laz files under input_dir, preserving subdirectory structure.
    Expects input_dir/{train,val,test}/*.laz or just input_dir/*.laz.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    laz_files = sorted(input_dir.rglob("*.laz"))
    if not laz_files:
        print(f"No .laz files found under {input_dir}")
        return

    print(f"Found {len(laz_files)} LAZ files. Target density: {density} pkt/m²\n")

    for laz_path in laz_files:
        rel = laz_path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            print(f"  Skipping {rel} (already exists)")
            continue

        downsample_laz(
            input_path=str(laz_path),
            output_path=str(out_path),
            density=density,
            cell_size=cell_size,
            seed=seed,
        )

    print(f"\nDone. Output at: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Downsample LAZ files to a target point density."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input directory containing .laz files (or subfolders train/val/test/).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory. Subdirectory structure is preserved.",
    )
    parser.add_argument(
        "--density",
        type=float,
        required=True,
        help="Target point density in points per square meter (e.g. 20).",
    )
    parser.add_argument(
        "--cell_size",
        type=float,
        default=1.0,
        help="Grid cell size in meters for spatial grouping (default: 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        density=args.density,
        cell_size=args.cell_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
