"""Per-tile precision/recall/IoU analysis on the bro test set."""
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path('/cluster/home/larshfle/superpoint_transformer_new')
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.utils import init_config
from src.transforms import *
from src.data import *
from src.datasets.bro import read_bro_las
from src.datasets.bro_config import BRO_NUM_CLASSES

CKPT = str(PROJECT_ROOT / 'logs/train/runs/2026-04-27_14-11-56/checkpoints/epoch_399.ckpt')
DATA_DIR = '/cluster/home/larshfle/datasets/bro'
SPLIT = 'test'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

cfg = init_config(overrides=[
    'experiment=semantic/bro',
    f'ckpt_path={CKPT}',
    'datamodule.mini=false',
    'datamodule.load_full_res_idx=true',
])
cfg.datamodule.data_dir = DATA_DIR

datamodule = hydra.utils.instantiate(cfg.datamodule)
datamodule.prepare_data()
datamodule.setup()
dataset = datamodule.test_dataset
print(f'Test tiles: {len(dataset)}')

model = hydra.utils.instantiate(cfg.model)
model = model._load_from_checkpoint(CKPT)
model = model.eval().to(device)
print('Model loaded.')

results = []

for i, tile_id in enumerate(dataset.cloud_ids):
    try:
        nag_i = dataset[i]
        nag_i = dataset.on_device_transform(nag_i.to(device))

        with torch.no_grad():
            out_i = model(nag_i)

        raw_pred = out_i.full_res_semantic_pred(
            super_index_level0_to_level1=nag_i[0].super_index,
            sub_level0_to_raw=nag_i[0].sub,
        ).cpu()

        raw_path_i = os.path.join(DATA_DIR, 'raw', SPLIT, f'{tile_id}.las')
        raw_labels_i = read_bro_las(raw_path_i).y.cpu()
        valid = raw_labels_i != BRO_NUM_CLASSES

        gt   = raw_labels_i[valid]
        pred = raw_pred[valid]

        tp = int(((gt == 1) & (pred == 1)).sum())
        fp = int(((gt == 0) & (pred == 1)).sum())
        fn = int(((gt == 1) & (pred == 0)).sum())
        n_bridge_gt = int((gt == 1).sum())

        recall    = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float('nan')

        results.append({
            'tile':        tile_id,
            'n_points':    int(valid.sum()),
            'n_bridge_gt': n_bridge_gt,
            'bridge_pct':  round(100 * n_bridge_gt / int(valid.sum()), 3) if valid.sum() > 0 else 0,
            'tp': tp, 'fp': fp, 'fn': fn,
            'recall':      round(recall, 4),
            'precision':   round(precision, 4),
            'iou':         round(iou, 4),
        })

        del nag_i, out_i, raw_pred
        torch.cuda.empty_cache()

        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(dataset.cloud_ids)} ...')

    except Exception as e:
        print(f'  ERROR {tile_id}: {e}')
        results.append({'tile': tile_id, 'iou': float('nan')})

df = pd.DataFrame(results).sort_values('iou', na_position='first')
out_csv = PROJECT_ROOT / 'logs/test_tile_analysis.csv'
df.to_csv(out_csv, index=False)

pos = df[df['n_bridge_gt'] > 0]
neg = df[df['n_bridge_gt'] == 0]

print(f'\n{"="*70}')
print(f'Totalt: {len(df)} tiles  |  Positive: {len(pos)}  |  Negative: {len(neg)}')
print(f'Positive mean IoU: {pos["iou"].mean():.4f}  |  Median: {pos["iou"].median():.4f}')
print(f'{"="*70}\n')

# Skriv markdown-tabell til fil
out_md = PROJECT_ROOT / 'logs/test_tile_analysis.md'
with open(out_md, 'w') as f:
    f.write('# Per-tile test analysis\n\n')
    f.write(f'Totalt: {len(df)} | Positive: {len(pos)} | Negative: {len(neg)}\n\n')
    f.write(f'Positive mean IoU: {pos["iou"].mean():.4f} | Median: {pos["iou"].median():.4f}\n\n')

    f.write('## Positive tiles (har bropunkter) — sortert på IoU\n\n')
    f.write('| Tile | n_bridge_gt | bridge_pct | recall | precision | iou |\n')
    f.write('|------|------------|------------|--------|-----------|-----|\n')
    for _, row in pos.iterrows():
        f.write(f"| {row['tile']} | {row['n_bridge_gt']} | {row['bridge_pct']} | {row['recall']} | {row['precision']} | {row['iou']} |\n")

    f.write('\n## Negative tiles (ingen bropunkter) — FP-rate\n\n')
    f.write('| Tile | n_points | fp_bridge_pred |\n')
    f.write('|------|----------|---------------|\n')
    for _, row in neg.iterrows():
        f.write(f"| {row['tile']} | {row.get('n_points','?')} | {int(row.get('fp', 0))} |\n")

print(f'Markdown-tabell lagret: {out_md}')
print(f'CSV lagret: {out_csv}')
