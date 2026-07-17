"""
Evaluate ground recall per tile and per dataset for all national_data tiles.
"""
import os, sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path('/cluster/home/larshfle/superpoint_transformer_new')
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import hydra
import torch
import numpy as np
import pandas as pd
import src.datasets.norway_combined_3class_config as gen_cfg
import src.datasets.norway_combined_3class as gen_mod

from src.utils import init_config
from src.transforms import *
from src.data import *
from src.datasets.norway_combined_3class_config import NORWAY_COMBINED_3CLASS_NUM_CLASSES

CKPT = '/cluster/home/larshfle/superpoint_transformer_new/logs/train/runs/2026-05-20_16-26-44/checkpoints/last.ckpt'
DATA_DIR = Path('/cluster/home/larshfle/datasets/national_data')
GROUND_CLASS = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Setup
tile_ids = [f.replace('.laz', '') for f in os.listdir(DATA_DIR / 'raw' / 'test')]
tile_ids = sorted(tile_ids)
gen_cfg.TILES = gen_mod.TILES = {'train': [], 'val': [], 'test': tile_ids}

cfg = init_config(overrides=[
    'experiment=semantic/norway_combined_3class',
    'datamodule=semantic/norway_combined_3class',
    f'paths.data_dir={DATA_DIR}',
    'datamodule.prepare_only_test=True',
])

model = hydra.utils.instantiate(cfg.model)
model = model._load_from_checkpoint(CKPT, map_location=device)
model = model.eval().to(device)

datamodule = hydra.utils.instantiate(cfg.datamodule)
datamodule.setup('test')
dataset = datamodule.test_dataset

print(f'Subtiles: {len(dataset.cloud_ids)}')

# Per-tile accumulators: {base_tile: [tp, fn]}
tile_stats = defaultdict(lambda: [0, 0])

for i, cid in enumerate(dataset.cloud_ids):
    print(f'[{i+1}/{len(dataset.cloud_ids)}] {cid}')
    try:
        nag = dataset[i]
        nag = dataset.on_device_transform(nag.to(device))

        with torch.no_grad():
            output = model(nag)
        nag[0].semantic_pred = output.voxel_semantic_pred(super_index=nag[0].super_index)

        gt   = nag[0].y.argmax(dim=-1).cpu().numpy()
        pred = nag[0].semantic_pred.cpu().numpy()
        valid = gt < NORWAY_COMBINED_3CLASS_NUM_CLASSES

        tp = int(((gt[valid] == GROUND_CLASS) & (pred[valid] == GROUND_CLASS)).sum())
        fn = int(((gt[valid] == GROUND_CLASS) & (pred[valid] != GROUND_CLASS)).sum())

        # Base tile name = everything before __TILE
        base = cid.split('__TILE')[0]
        tile_stats[base][0] += tp
        tile_stats[base][1] += fn
    except Exception as e:
        print(f'  SKIP: {e}')

# Build per-tile results
rows = []
for tile, (tp, fn) in sorted(tile_stats.items()):
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    dataset_name = tile.split('_')[0]  # alta, malselv, tromsoya, arendal
    rows.append({'dataset': dataset_name, 'tile': tile,
                 'ground_recall': f'{100*recall:.1f}%',
                 'tp': tp, 'fn': fn})

df = pd.DataFrame(rows)
print('\n=== Ground Recall per Tile ===')
print(df[['dataset', 'tile', 'ground_recall']].to_string(index=False))

print('\n=== Ground Recall per Dataset ===')
for ds, grp in df.groupby('dataset'):
    tp_tot = grp['tp'].sum()
    fn_tot = grp['fn'].sum()
    recall = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else float('nan')
    print(f'  {ds:<12} {100*recall:.1f}%  ({len(grp)} tiles)')

# Save CSV
out = PROJECT_ROOT / 'logs' / 'ground_recall_national.csv'
df.to_csv(out, index=False)
print(f'\nSaved: {out}')
