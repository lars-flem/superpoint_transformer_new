"""
Per-subtile evaluation for the norway_combined_3class model.
Runs on CPU. Writes per-tile results to txt/csv and prints region summary at the end.
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

from src.utils import init_config
from src.transforms import *
from src.data import *
from src.datasets.norway_combined_3class_config import NORWAY_COMBINED_3CLASS_NUM_CLASSES

# ── Settings ─────────────────────────────────────────────────────────────
CKPT_PATH = str(PROJECT_ROOT / 'logs/train/runs/2026-05-20_16-26-44/checkpoints/last.ckpt')
DATA_DIR  = '/cluster/home/larshfle/datasets/norway_combined_3class'
OUT_TXT   = str(PROJECT_ROOT / 'logs/eval_per_tile_test.txt')
OUT_CSV   = str(PROJECT_ROOT / 'logs/eval_per_tile_test.csv')
DEVICE    = 'cpu'
# ─────────────────────────────────────────────────────────────────────────

print(f'Device: {DEVICE}')
print(f'Checkpoint: {CKPT_PATH}\n')

cfg = init_config(overrides=[
    'experiment=semantic/norway_combined_3class',
    f'ckpt_path={CKPT_PATH}',
])
cfg.datamodule.data_dir = DATA_DIR

datamodule = hydra.utils.instantiate(cfg.datamodule)
datamodule.prepare_data()
datamodule.setup()
dataset = datamodule.test_dataset
CLASS_NAMES = dataset.class_names[:dataset.num_classes]
NUM_CLASSES = NORWAY_COMBINED_3CLASS_NUM_CLASSES
print(f'Test tiles: {len(dataset.cloud_ids)}  |  Classes: {CLASS_NAMES}\n')

model = hydra.utils.instantiate(cfg.model)
model = model._load_from_checkpoint(CKPT_PATH, map_location=DEVICE)
model = model.eval().to(DEVICE)
print('Model loaded.\n')

# Accumulated TP/FP/FN per region per class
region_counts = defaultdict(lambda: np.zeros((NUM_CLASSES, 3), dtype=np.int64))
global_counts = np.zeros((NUM_CLASSES, 3), dtype=np.int64)

header = f"{'tile':<50} {'region':<12} {'mIoU':>6}  " + \
         '  '.join(f'{n}(IoU/P/R)' for n in CLASS_NAMES)

with open(OUT_TXT, 'w') as f:
    f.write(header + '\n')
    f.write('-' * len(header) + '\n')

for i, tile_id in enumerate(dataset.cloud_ids):
    region = tile_id.split('_')[0]
    print(f'[{i+1}/{len(dataset.cloud_ids)}] {tile_id}')

    try:
        nag = dataset[i]
        nag = dataset.on_device_transform(nag.to(DEVICE))

        with torch.no_grad():
            output = model(nag)

        pred  = output.voxel_semantic_pred(super_index=nag[0].super_index).cpu().numpy()
        gt    = nag[0].y.argmax(dim=-1).cpu().numpy()
        valid = gt < NUM_CLASSES
        del nag, output

        tile_metrics = []
        csv_line = f'{tile_id},{region}'
        for c in range(NUM_CLASSES):
            tp = int(((gt[valid]==c) & (pred[valid]==c)).sum())
            fp = int(((gt[valid]!=c) & (pred[valid]==c)).sum())
            fn = int(((gt[valid]==c) & (pred[valid]!=c)).sum())
            iou  = tp/(tp+fp+fn) if (tp+fp+fn)>0 else float('nan')
            prec = tp/(tp+fp)    if (tp+fp)>0    else float('nan')
            rec  = tp/(tp+fn)    if (tp+fn)>0    else float('nan')
            tile_metrics.append((iou, prec, rec))
            region_counts[region][c] += [tp, fp, fn]
            global_counts[c]         += [tp, fp, fn]
            csv_line += f',{tp},{fp},{fn}'

        miou = np.nanmean([m[0] for m in tile_metrics])
        per_class = '  '.join(f'{100*iou:5.1f}/{100*prec:5.1f}/{100*rec:5.1f}'
                              for iou, prec, rec in tile_metrics)
        txt_line = f'{tile_id:<50} {region:<12} {100*miou:5.1f}%  {per_class}\n'
        print(f'  mIoU={100*miou:.1f}%  ' +
              '  '.join(f'{CLASS_NAMES[c]}={100*tile_metrics[c][0]:.1f}%' for c in range(NUM_CLASSES)))

    except Exception as e:
        print(f'  ERROR: {e}')
        txt_line = f'{tile_id:<50} {region:<12}   ERR  {e}\n'
        csv_line = f'{tile_id},{region}' + ',0,0,0' * NUM_CLASSES

    with open(OUT_TXT, 'a') as f:
        f.write(txt_line)
    with open(OUT_CSV, 'a') as f:
        f.write(csv_line + '\n')

# ── Per-region summary ────────────────────────────────────────────────────
def print_and_write_summary(f, counts_dict, global_counts, CLASS_NAMES):
    NUM_CLASSES = len(CLASS_NAMES)
    for region in ['bergen2020', 'bergen2022', 'oslo', 'viken']:
        rc = counts_dict.get(region)
        if rc is None:
            continue
        lines = [f'\n{"="*60}', f'{region.upper()}',
                 f'  {"klasse":<12}  {"IoU":>6}  {"Precision":>9}  {"Recall":>7}',
                 f'  {"-"*42}']
        ious = []
        for c, name in enumerate(CLASS_NAMES):
            tp, fp, fn = rc[c]
            iou  = tp/(tp+fp+fn) if (tp+fp+fn)>0 else float('nan')
            prec = tp/(tp+fp)    if (tp+fp)>0    else float('nan')
            rec  = tp/(tp+fn)    if (tp+fn)>0    else float('nan')
            ious.append(iou)
            lines.append(f'  {name:<12}  {100*iou:5.1f}%  {100*prec:9.1f}%  {100*rec:7.1f}%')
        lines.append(f'  {"mIoU":<12}  {100*np.nanmean(ious):5.1f}%')
        block = '\n'.join(lines)
        print(block)
        f.write(block + '\n')

    lines = [f'\n{"="*60}', 'TOTAL',
             f'  {"klasse":<12}  {"IoU":>6}  {"Precision":>9}  {"Recall":>7}',
             f'  {"-"*42}']
    ious = []
    for c, name in enumerate(CLASS_NAMES):
        tp, fp, fn = global_counts[c]
        iou  = tp/(tp+fp+fn) if (tp+fp+fn)>0 else float('nan')
        prec = tp/(tp+fp)    if (tp+fp)>0    else float('nan')
        rec  = tp/(tp+fn)    if (tp+fn)>0    else float('nan')
        ious.append(iou)
        lines.append(f'  {name:<12}  {100*iou:5.1f}%  {100*prec:9.1f}%  {100*rec:7.1f}%')
    lines.append(f'  {"mIoU":<12}  {100*np.nanmean(ious):5.1f}%')
    block = '\n'.join(lines)
    print(block)
    f.write(block + '\n')

with open(OUT_TXT, 'a') as f:
    print('\n\nPer-region summary (aggregert TP/FP/FN):')
    f.write('\n\nPer-region summary (aggregert TP/FP/FN):\n')
    print_and_write_summary(f, dict(region_counts), global_counts, CLASS_NAMES)

print(f'\nFerdig. Resultater: {OUT_TXT}')
