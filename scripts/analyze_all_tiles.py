"""Per-tile precision/recall/IoU for val + test, skriver MD for manuell gjennomgang."""
import os
import sys
from pathlib import Path

import hydra
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

CKPT     = str(PROJECT_ROOT / 'logs/train/runs/2026-05-05_05-24-36/checkpoints/last.ckpt')
DATA_DIR = '/cluster/home/larshfle/datasets/bro'

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

model = hydra.utils.instantiate(cfg.model)
model = model._load_from_checkpoint(CKPT)
model = model.eval().to(device)
print('Model loaded.')


def run_split(dataset, split_name):
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

            raw_path_i = os.path.join(DATA_DIR, 'raw', split_name, f'{tile_id}.las')
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
                print(f'  [{split_name}] {i+1}/{len(dataset.cloud_ids)} ...')

        except Exception as e:
            print(f'  ERROR {tile_id}: {e}')
            results.append({'tile': tile_id, 'n_bridge_gt': 0, 'iou': float('nan'),
                            'recall': float('nan'), 'precision': float('nan'),
                            'bridge_pct': float('nan'), 'fp': 0})

    return pd.DataFrame(results).sort_values('iou', na_position='first')


def write_split_md(f, df, split_name):
    pos = df[df['n_bridge_gt'] > 0]
    neg = df[df['n_bridge_gt'] == 0]

    f.write(f'## {split_name.upper()} — {len(df)} tiles '
            f'({len(pos)} positive, {len(neg)} negative)\n\n')
    f.write(f'Mean IoU (positive): **{pos["iou"].mean():.4f}** | '
            f'Median: **{pos["iou"].median():.4f}**\n\n')

    f.write('### Positive tiles — sortert på IoU (verst øverst)\n\n')
    f.write('| Tile | n_bridge_gt | bridge_pct% | recall | precision | iou |\n')
    f.write('|------|------------|-------------|--------|-----------|-----|\n')
    for _, row in pos.iterrows():
        f.write(f"| {row['tile']} | {row['n_bridge_gt']} | {row['bridge_pct']} "
                f"| {row['recall']} | {row['precision']} | {row['iou']} |\n")

    f.write('\n### Negative tiles — FP-prediksjoner\n\n')
    f.write('| Tile | n_points | fp_bridge_pred |\n')
    f.write('|------|----------|---------------|\n')
    for _, row in neg.sort_values('fp', ascending=False).iterrows():
        f.write(f"| {row['tile']} | {row.get('n_points','?')} | {int(row.get('fp', 0))} |\n")
    f.write('\n---\n\n')


# Kjør begge splits
print('\n=== VAL ===')
df_val = run_split(datamodule.val_dataset, 'val')

print('\n=== TEST ===')
df_test = run_split(datamodule.test_dataset, 'test')

# Lagre CSV
df_val.to_csv(PROJECT_ROOT / 'logs/val_tile_analysis.csv', index=False)
df_test.to_csv(PROJECT_ROOT / 'logs/test_tile_analysis.csv', index=False)

# Lagre MD
out_md = PROJECT_ROOT / 'logs/tile_quality_review.md'
with open(out_md, 'w') as f:
    f.write('# Tile quality review — val + test\n\n')
    f.write('Bruk denne filen til å identifisere tiles som bør fjernes fra datasettet.\n\n')
    write_split_md(f, df_val, 'val')
    write_split_md(f, df_test, 'test')

print(f'\nFerdig! Åpne: logs/tile_quality_review.md')
