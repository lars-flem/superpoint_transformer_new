# Setting Up a New Dataset in SPT

This guide documents how to set up and use the SuperPoint Transformer (SPT) framework with a custom dataset from scratch, based on our experience with Norwegian ALS data. It supplements the original repository documentation with practical recommendations.

---

## 0. Environment Setup

Before anything else, create the conda environment and install dependencies.
This only needs to be done once per machine. The repository ships with an
`install.sh` script that handles the full setup — clone the repo and run it:

```bash
# Clone the repository
git clone https://github.com/lars-flem/superpoint_transformer_new.git
cd superpoint_transformer_new

# Run the install script (creates a conda env named 'spt' with Python 3.8,
# PyTorch 2.2.0, and all SPT dependencies including FRNN built from source)
./install.sh

# Optional: also install TorchSparse
./install.sh with_torchsparse
```

The script expects CUDA 11.8 or 12.1 to be installed and discoverable via `nvcc`,
and conda to live at `~/miniconda3` or `~/anaconda3` (it will prompt for a path
otherwise). When it finishes, activate the environment with `conda activate spt`.

> **If a specific package fails to install,** the script will abort partway
> through but the conda environment will already exist. Just activate it
> (`conda activate spt`) and install the failing package manually with `pip
> install <package>`; the rest of the installed dependencies are preserved. This
> is common for packages with network-flaky downloads or compilation steps
> (e.g. `FRNN`, `torchsparse`, the PyG wheels).

> **On SLURM clusters (e.g. NTNU Idun):** load matching CUDA and GCC modules
> before running the installer:
> ```bash
> module load CUDA/11.8.0
> module load GCC/11.3.0
> ./install.sh
> ```

Verify the environment works:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Overview of Files to Create

For a new dataset called `mydataset`, you need to create or modify:

```
src/datasets/mydataset_config.py      # tile split + class mapping + colours
src/datasets/mydataset.py             # dataset class (reads raw files, maps labels)
src/datamodules/mydataset.py          # thin datamodule wrapper
configs/datamodule/semantic/mydataset.yaml    # preprocessing + training parameters
configs/experiment/semantic/mydataset.yaml   # training run config (epochs, lr, logging)
scripts/train_mydataset.sbatch               # SLURM job script
```

---

## Step 1 — Organise raw data

SPT expects raw files under `{data_dir}/raw/{split}/`:

```
datasets/mydataset/
  raw/
    train/   tile_001.laz  tile_002.laz ...
    val/     tile_101.laz  ...
    test/    tile_201.laz  ...
```

Use symlinks if tiles already exist elsewhere:
```bash
ln -sf /path/to/actual/tile_001.laz datasets/mydataset/raw/train/tile_001.laz
```

For large datasets with tiles spread across multiple source directories,
write a setup script (see `scripts/setup_bergen2020.sh` for an example).

---

## Step 2 — Config file (`mydataset_config.py`)

```python
import numpy as np

TILES = {
    "train": ["tile_001", "tile_002", ...],
    "val":   ["tile_101", ...],
    "test":  ["tile_201", ...],
}

# Map LAS classification codes → training IDs
# Fill everything with ignored by default
ID2TRAINID = np.full(256, NUM_CLASSES, dtype=np.int64)
ID2TRAINID[2] = 0   # Ground
ID2TRAINID[5] = 1   # High vegetation → not_ground
ID2TRAINID[6] = 2   # Building

NUM_CLASSES = 3
CLASS_NAMES  = ["ground", "not_ground", "building", "ignored"]
CLASS_COLORS = [
    [140,  90,  60],   # ground      (brown)
    [180, 180, 180],   # not_ground  (grey)
    [255, 210,   0],   # building    (yellow)
    [  0,   0,   0],   # ignored     (black)
]
```

**Tips:**
- Always assign a default ignored class equal to `NUM_CLASSES`.
- Map classes like noise (7) and unclassified (1) to ignored.
---

## Step 3 — Dataset class (`mydataset.py`)

Minimal implementation reading LAZ/LAS files:

```python
import laspy, numpy as np, torch
from src.datasets import BaseDataset
from src.data import Data
import src.datasets.mydataset_config as cfg

class MyDatasetALS(BaseDataset):
    def __init__(self, *args, rgb: bool = False, **kwargs):
        self.rgb = rgb
        super().__init__(*args, **kwargs)

    @property
    def data_subdir_name(self): return ""

    @property
    def class_names(self): return cfg.CLASS_NAMES

    @property
    def num_classes(self): return cfg.NUM_CLASSES

    @property
    def stuff_classes(self): return list(range(self.num_classes))

    @property
    def class_colors(self): return cfg.CLASS_COLORS

    @property
    def all_base_cloud_ids(self): return cfg.TILES

    def download_dataset(self):
        raise RuntimeError("No auto-download. Organise raw files manually.")

    def read_single_raw_cloud(self, raw_cloud_path):
        las = laspy.read(raw_cloud_path)
        data = Data()
        pos = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        pos = torch.from_numpy(pos)
        data.pos = pos - pos[0]
        data.pos_offset = pos[0]
        cls = np.asarray(las.classification, dtype=np.int64)
        data.y = torch.from_numpy(cfg.ID2TRAINID[cls]).long()
        if "intensity" in las.point_format.dimension_names:
            inten = np.asarray(las.intensity, dtype=np.float32)
            data.intensity = torch.from_numpy(inten / max(inten.max(), 1.0))
        return data

    def id_to_relative_raw_path(self, id):
        base = self.id_to_base_id(id)
        for split in ("train", "val", "test"):
            if base in cfg.TILES[split]:
                return f"{split}/{base}.laz"
        raise ValueError(f"Unknown tile: {id}")

    def processed_to_raw_path(self, processed_path):
        cloud_id = processed_path.split("/")[-1].replace(".h5", "")
        base = self.id_to_base_id(cloud_id)
        for split in ("train", "val", "test"):
            if base in cfg.TILES[split]:
                return f"{self.raw_dir}/{split}/{base}.laz"
        raise ValueError(f"Unknown tile: {cloud_id}")


class MiniMyDatasetALS(MyDatasetALS):
    @property
    def all_base_cloud_ids(self):
        return {k: v[:2] for k, v in cfg.TILES.items()}
```

---

## Step 4 — Datamodule (`src/datamodules/mydataset.py`)

```python
from src.datamodules.base import BaseDataModule
from src.datasets.mydataset import MyDatasetALS, MiniMyDatasetALS

class MyDatasetDataModule(BaseDataModule):
    _DATASET_CLASS      = MyDatasetALS
    _MINIDATASET_CLASS  = MiniMyDatasetALS
```

---

## Step 5 — Datamodule config (`configs/datamodule/semantic/mydataset.yaml`)

Key parameters and recommendations:

```yaml
# @package datamodule
defaults:
  - /datamodule/semantic/default.yaml

_target_: src.datamodules.mydataset.MyDatasetDataModule

data_dir: /path/to/datasets/mydataset
num_classes: 3
stuff_classes: [0, 1, 2]
trainval: False
val_on_test: False

# ── Tiling ────────────────────────────────────────────────────────────────────
# Subdivides each tile into an n×n grid of subtiles.
# Larger n = more training samples, but more memory per tile.
# Recommendation:
#   - Large tiles (800×600 m, >5M pts): xy_tiling: 3  (9 subtiles)
#   - Medium tiles (400×400 m):         xy_tiling: 2  (4 subtiles)
#   - Small tiles (bridge buffers etc): xy_tiling: null
# Note: higher xy_tiling requires more RAM during preprocessing.
xy_tiling: 2
min_points_per_subtile: 1000   # discard near-empty subtiles

# ── Voxelisation ──────────────────────────────────────────────────────────────
# 0.2 m is a good default for 5–15 pts/m².
# Use 0.1 m for high-density data (≥ 20 pts/m²).
voxel: 0.2

# ── Partition features (used by Cut-Pursuit to form superpoints) ──────────────
partition_hf:
  - "linearity"
  - "planarity"
  - "scattering"
  - "elevation"
# Add "rgb" here if your data has colour and you want colour-coherent superpoints.

# ── Point features (input to the neural network) ──────────────────────────────
point_hf:
  - "intensity"
  - "linearity"
  - "planarity"
  - "scattering"
  - "verticality"
  - "elevation"
# Do NOT add "rgb" here unless you want the model to depend on colour at inference.

# ── Preprocessing parameters (rarely need changing) ───────────────────────────
knn: 25
knn_r: 10
knn_step: -1
knn_min_search: 10
ground_model: "ransac"
ground_threshold: 5
ground_xy_grid: 1
ground_scale: 20
pcp_regularization: [0.1, 0.2, 0.3]   # coarser = larger superpoints
pcp_spatial_weight: [1e-1, 1e-2, 1e-3]
pcp_edge_reduce: "mean"
pcp_cutoff: [10, 30, 100]
pcp_k_adjacency: 10
pcp_w_adjacency: 1
pcp_iterations: 15
graph_k_min: 1
graph_k_max: 30
graph_gap: [5, 30, 30]
graph_se_ratio: 0.3
graph_se_min: 20
graph_cycles: 3
graph_margin: 0.5
graph_chunk: [1e6, 1e5, 1e5]

# ── Batch sampling ────────────────────────────────────────────────────────────
# sample_graph_r: radius (m) of each spherical training subgraph.
# Larger = more context, more memory. 50 m is a good default.
sample_segment_ratio: 0.2
sample_segment_by_size: True
sample_segment_by_class: False   # set True for severe class imbalance
sample_point_min: 32
sample_point_max: 128
sample_graph_r: 50
sample_graph_k: 4
sample_graph_max_nodes: 10000
sample_graph_disjoint: True
sample_graph_cylindrical: True
sample_edge_n_min: -1
sample_edge_n_max: -1

# ── Augmentations ─────────────────────────────────────────────────────────────
pos_jitter: 0.05
tilt_n_rotate_phi: 0.1
tilt_n_rotate_theta: 180
anisotropic_scaling: 0.2
node_feat_jitter: 0
h_edge_feat_jitter: 0
v_edge_feat_jitter: 0
node_feat_drop: 0
h_edge_feat_drop: 0.3
v_edge_feat_drop: 0
node_row_drop: 0
h_edge_row_drop: 0
v_edge_row_drop: 0
drop_to_mean: False

# ── RGB (set rgb: False if data has no colour) ────────────────────────────────
rgb: False
rgb_jitter: 0
rgb_autocontrast: 0.5
rgb_drop: 0.3
```

---

## Step 6 — Experiment config (`configs/experiment/semantic/mydataset.yaml`)

```yaml
# @package _global_
# python src/train.py experiment=semantic/mydataset

defaults:
  - override /datamodule: semantic/mydataset.yaml
  - override /model: semantic/spt-2.yaml
  - override /trainer: gpu.yaml

trainer:
  max_epochs: 400

model:
  optimizer:
    lr: 0.01
    weight_decay: 1e-4

logger:
  wandb:
    project: "spt_mydataset"
    name: "SPT-64"
```

---

## Step 7 — Start training

```bash
conda activate spt
cd /path/to/superpoint_transformer_new

python src/train.py \
    experiment=semantic/mydataset \
    datamodule=semantic/mydataset \
    paths.data_dir=/path/to/datasets/mydataset \
    datamodule.dataloader.num_workers=8 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.model_checkpoint.save_last=True \
    callbacks.model_checkpoint.every_n_epochs=5 \
    callbacks.model_checkpoint.monitor=null \
    test=True
```

### Force re-preprocessing

Processed `.h5` files are cached under `{data_dir}/processed/` keyed by a hash of
the preprocessing parameters. Changing `voxel`, `pcp_regularization`, `xy_tiling`,
or `num_classes` automatically triggers re-preprocessing on the next run.

To force re-preprocessing manually (e.g. after fixing a bug in the dataset class):

```bash
rm -rf /path/to/datasets/mydataset/processed/
```

Or if you want to keep the old preprocessed files:

```bash
mv /path/to/datasets/mydataset/processed/ /path/to/datasets/mydataset/processed_old/
```

---

## Practical Recommendations

### XY tiling
- `xy_tiling: 3` for large, dense tiles (e.g. 800×600 m at ≥10 pts/m²). Produces
  9 subtiles, roughly tripling the number of training samples. Requires ~3× more RAM
  during preprocessing. The `min_points_per_subtile` parameter discards sparse subtiles
  with too few points to avoid degenerate preprocessing.
- `xy_tiling: 2` for medium tiles. Good balance.
- `xy_tiling: null` for small specialised tiles (e.g. bridge buffers, small test areas).

### Voxel size
- 0.2 m: default for 5–15 pts/m²
- 0.1 m: high-density data (≥20 pts/m²) — retains more detail but increases processing time

### Re-preprocessing
Processed `.h5` files are cached by hash. Changes to any preprocessing parameter
(`voxel`, `pcp_regularization`, `xy_tiling`, etc.) automatically trigger re-preprocessing
on the next run. To force it manually, delete `{data_dir}/processed/`.

### Class imbalance
- `sample_segment_by_class: True` oversamples minority-class segments during training.
  Useful when one class makes up <1% of points (e.g. bridges). Can cause
  over-prediction — monitor precision as well as recall.
- `weighted_loss_smooth: 'log'` applies log-weighted class loss. Gentler than `'sqrt'`.

### RGB
- Include `"rgb"` in `partition_hf` to form colour-coherent superpoints — helps in
  urban scenes with distinct building/vegetation colours.
- Do **not** include `"rgb"` in `point_hf` if you want the model to work on datasets
  without colour, or if early experiments show little improvement from colour features.

### Memory
GPU memory usage goes up with higher xy_tiling, tile_size, and point_density.

If you hit OutOfMemory errors, reduce `graph_chunk`, lower `xy_tiling`, increase the
`voxel` size to reduce point count, or use a GPU with more memory.
