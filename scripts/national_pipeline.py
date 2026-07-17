"""
Two-stage inference pipeline:
  1. General 3-class model  (ground / not_ground / building)
  2. Bridge model           (background / bridge)
     - Bridge predictions overwrite general predictions,
       EXCEPT where the general model predicted building.
"""
import argparse
import os
import sys
from pathlib import Path

import hydra
import laspy
import numpy as np
import torch

PROJECT_ROOT = Path('/cluster/home/larshfle/superpoint_transformer_new')
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.utils import init_config
from src.transforms import *
from src.data import *
import src.datasets.bro_config as bro_cfg
import src.datasets.bro as bro_mod
import src.datasets.norway_combined_3class_config as gen_cfg
import src.datasets.norway_combined_3class as gen_mod
from src.datasets.norway_combined_3class_config import (
    ID2TRAINID as GEN_MAP,
    NORWAY_COMBINED_3CLASS_NUM_CLASSES as GEN_NC,
)



# ── Class IDs in final output ────────────────────────────────────────────────
OUT_GROUND     = 0
OUT_NOT_GROUND = 1
OUT_BUILDING   = 2
OUT_BRIDGE     = 3
OUT_IGNORED    = 255

OUT_CLASS_NAMES  = ['ground', 'not_ground', 'building', 'bridge']
OUT_CLASS_COLORS = [
    [140, 90,  60],   # ground      brown
    [180, 180, 180],  # not_ground  grey
    [255, 210,   0],  # building    yellow
    [  0,  80, 255],  # bridge      blue
]


def setup_general_tmp(raw_laz: Path, tile_id: str) -> Path:
    """Set up a temp dir for the general model with a single unseen tile."""
    tmp = Path(f'/tmp/gen_pipeline_{tile_id}')
    for split in ('train', 'val', 'test'):
        (tmp / 'raw' / split).mkdir(parents=True, exist_ok=True)
    laz_out = tmp / 'raw' / 'test' / f'{tile_id}.laz'
    if not laz_out.exists():
        laz_out.symlink_to(raw_laz.resolve())
    new_tiles = {'train': [], 'val': [], 'test': [tile_id]}
    gen_cfg.TILES = new_tiles
    gen_mod.TILES = new_tiles
    return tmp


def setup_bro_tmp(raw_laz: Path, tile_id: str) -> Path:
    """Copy a LAZ file into a temp bro directory and patch TILES."""
    tmp = Path(f'/tmp/bro_pipeline_{tile_id}')
    las_out = tmp / 'raw' / 'test' / f'{tile_id}.las'
    las_out.parent.mkdir(parents=True, exist_ok=True)
    (tmp / 'raw' / 'train').mkdir(parents=True, exist_ok=True)
    (tmp / 'raw' / 'val').mkdir(parents=True, exist_ok=True)

    if not las_out.exists():
        print(f'  Converting {raw_laz.name} → {las_out}')
        import laspy as _lp
        _lp.read(str(raw_laz)).write(str(las_out))

    new_tiles = {'train': [], 'val': [], 'test': [tile_id]}
    bro_cfg.TILES = new_tiles
    bro_mod.TILES = new_tiles
    return tmp


def load_model(experiment, ckpt_path, device, extra_overrides=None):
    overrides = [
        f'experiment={experiment}',
        f'ckpt_path={ckpt_path}',
        'datamodule.mini=false',
        'datamodule.load_full_res_idx=true',
        f'datamodule.pre_transform.0.params.device={device}',
    ]
    if extra_overrides:
        overrides += extra_overrides
    cfg = init_config(overrides=overrides)
    model = hydra.utils.instantiate(cfg.model)
    model = model._load_from_checkpoint(ckpt_path, map_location=device)
    model = model.eval().to(device)
    model.net.store_features = True
    return cfg, model


def run_inference(dataset, tile_id, model, device):
    """Return voxel-level predictions for a tile (handles xy_tiling)."""
    subtile_ids = [
        i for i, cid in enumerate(dataset.cloud_ids)
        if cid.startswith(tile_id)
    ]
    if not subtile_ids:
        raise ValueError(f'Tile {tile_id!r} not found in dataset.')

    all_pos, all_pred = [], []
    for idx in subtile_ids:
        nag = dataset[idx]
        nag = dataset.on_device_transform(nag.to(device))
        with torch.no_grad():
            out = model(nag)
        pred = out.voxel_semantic_pred(super_index=nag[0].super_index).cpu()
        pos  = (nag[0].pos + nag[0].pos_offset.to(nag[0].pos.device)).cpu()
        all_pos.append(pos)
        all_pred.append(pred)
        del nag, out
        torch.cuda.empty_cache()

    return torch.cat(all_pos), torch.cat(all_pred)


def compute_scores(pred, gt, num_classes, class_names):
    rows = []
    for c in range(num_classes):
        valid = gt < num_classes
        tp = int(((gt[valid] == c) & (pred[valid] == c)).sum())
        fp = int(((gt[valid] != c) & (pred[valid] == c)).sum())
        fn = int(((gt[valid] == c) & (pred[valid] != c)).sum())
        iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float('nan')
        prec = tp / (tp + fp)       if (tp + fp) > 0       else float('nan')
        rec  = tp / (tp + fn)       if (tp + fn) > 0       else float('nan')
        rows.append({
            'class': class_names[c],
            'iou':   round(iou,  4),
            'prec':  round(prec, 4),
            'rec':   round(rec,  4),
            'gt_pts': tp + fn,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Two-stage SPT inference pipeline')
    parser.add_argument('--tile',          required=True,  help='Tile ID (e.g. bro_001709)')
    parser.add_argument('--general_ckpt',  required=True,  help='Path to general 3-class checkpoint')
    parser.add_argument('--general_data',  required=True,  help='Data dir for general model')
    parser.add_argument('--bridge_ckpt',   required=True,  help='Path to bridge model checkpoint')
    parser.add_argument('--bridge_data',   required=True,  help='Data dir for bridge model')
    parser.add_argument('--raw_laz',       default=None,   help='Path to raw LAZ for bridge model (for non-bro tiles)')
    parser.add_argument('--split',         default='test', help='train/val/test')
    parser.add_argument('--out_dir',       default=str(PROJECT_ROOT / 'output_pipeline'),
                                           help='Output directory')
    parser.add_argument('--no_building_mask', action='store_true',
                        help='Disable building mask (allow bridge to overwrite building)')
    parser.add_argument('--no_las',  action='store_true', help='Skip saving classified point cloud')
    parser.add_argument('--laz',     action='store_true', help='Save as LAZ instead of LAS (compressed)')
    parser.add_argument('--bridge_xy_tiling', type=int, default=3,
                        help='xy_tiling for bridge model (default 3 = 9 subtiles)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: General 3-class model ───────────────────────────────────────
    print('\n=== Stage 1: General model ===')
    cfg_gen, model_gen = load_model('semantic/norway_combined_3class', args.general_ckpt, device)
    gen_data = args.general_data
    if args.raw_laz:
        print(f'  Setting up temp general dir for {args.tile}')
        gen_data = str(setup_general_tmp(Path(args.raw_laz), args.tile))
    cfg_gen.datamodule.data_dir = gen_data
    dm_gen = hydra.utils.instantiate(cfg_gen.datamodule)
    dm_gen.prepare_data()
    dm_gen.setup()
    dataset_gen = getattr(dm_gen, f'{args.split}_dataset')

    pos, general_pred = run_inference(dataset_gen, args.tile, model_gen, device)
    print(f'  Points: {len(pos):,}  |  Unique preds: {general_pred.unique().tolist()}')
    # ── Stage 2: Bridge model ─────────────────────────────────────────────────
    print('\n=== Stage 2: Bridge model ===')

    # If a raw LAZ is provided, set up a temp bro directory
    bro_tile_id = args.tile
    bro_data    = args.bridge_data
    bro_split   = args.split
    if args.raw_laz:
        print(f'  Using temp bro dir for {args.tile}')
        bro_data    = str(setup_bro_tmp(Path(args.raw_laz), args.tile))
        bro_tile_id = args.tile
        bro_split   = 'test'

    cfg_bro, model_bro = load_model('semantic/bro', args.bridge_ckpt, device,
                                     extra_overrides=[f'datamodule.xy_tiling={args.bridge_xy_tiling}'])
    cfg_bro.datamodule.data_dir = bro_data
    dm_bro = hydra.utils.instantiate(cfg_bro.datamodule)
    dm_bro.prepare_data()
    dm_bro.setup()
    dataset_bro = getattr(dm_bro, f'{bro_split}_dataset')

    _, bridge_pred = run_inference(dataset_bro, bro_tile_id, model_bro, device)
    print(f'  Bridge predictions: {(bridge_pred == 1).sum().item():,}')
    del model_bro, dm_bro
    torch.cuda.empty_cache()

    # ── Merge predictions ─────────────────────────────────────────────────────
    print('\n=== Merging predictions ===')
    # Map general 3-class → final output IDs
    gen_map = {0: OUT_GROUND, 1: OUT_NOT_GROUND, 2: OUT_BUILDING}
    final = torch.full((len(pos),), OUT_IGNORED, dtype=torch.long)
    for src, dst in gen_map.items():
        final[general_pred == src] = dst

    # Overwrite with bridge predictions (except where general says building)
    is_bridge   = bridge_pred == 1
    is_building = general_pred == 2

    if args.no_building_mask:
        mask = is_bridge
    else:
        mask = is_bridge & ~is_building

    final[mask] = OUT_BRIDGE
    print(f'  Final bridge points: {(final == OUT_BRIDGE).sum().item():,}')
    if not args.no_building_mask:
        blocked = (is_bridge & is_building).sum().item()
        print(f'  Bridge predictions blocked by building mask: {blocked:,}')

    # ── Export LAS ────────────────────────────────────────────────────────────
    pos_np = pos.numpy().astype(np.float64)
    final_np = final.numpy().astype(np.uint8)

    colors = np.array(OUT_CLASS_COLORS + [[50, 50, 50]], dtype=np.uint16)  # last = ignored
    color_idx = np.clip(final_np, 0, len(colors) - 1)
    rgb = (colors[color_idx] * 256).astype(np.uint16)

    header = laspy.LasHeader(point_format=2, version='1.2')
    las_out = laspy.LasData(header=header)
    las_out.x = pos_np[:, 0]
    las_out.y = pos_np[:, 1]
    las_out.z = pos_np[:, 2]
    las_out.red   = rgb[:, 0]
    las_out.green = rgb[:, 1]
    las_out.blue  = rgb[:, 2]
    las_out.classification = final_np

    if not args.no_las:
        ext = '.laz' if args.laz else '.las'
        out_las = out_dir / f'{args.tile}_pipeline{ext}'
        las_out.write(str(out_las))
        print(f'\nPoint cloud saved: {out_las}')

    # ── Class distribution ────────────────────────────────────────────────────
    print('\n=== Final class distribution ===')
    for cid, name in enumerate(OUT_CLASS_NAMES):
        n = (final == cid).sum().item()
        pct = 100 * n / len(final)
        print(f'  {name:<12}: {n:>10,}  ({pct:.2f}%)')

    # ── Scores vs ground truth ────────────────────────────────────────────────
    print('\n=== Scores vs ground truth ===')

    # Read GT from raw LAZ
    raw_laz_path = Path(args.raw_laz) if args.raw_laz else None
    score_lines = []

    if raw_laz_path and raw_laz_path.exists():
        raw_las = laspy.read(str(raw_laz_path))
        raw_cls = np.asarray(raw_las.classification, dtype=np.int64)

        # Build 4-class GT: ground=0, not_ground=1, building=2, bridge=3
        gt4 = np.full(len(raw_cls), 255, dtype=np.int64)
        gen_id = GEN_MAP[raw_cls]
        gt4[gen_id == 0] = OUT_GROUND
        gt4[gen_id == 1] = OUT_NOT_GROUND
        gt4[gen_id == 2] = OUT_BUILDING
        gt4[raw_cls == 17] = OUT_BRIDGE  # bridge overwrites not_ground in GT

        # The final predictions are voxel-level; GT is full-res — compute on GT side
        # For simplicity, aggregate final per class and compare with GT counts
        # (Full alignment requires full_res_pred which needs sub index)
        # Instead: report per-class GT counts and predicted counts
        header_line = f'\n{"Class":<14} {"GT pts":>10} {"Pred pts":>10}'
        print(header_line)
        score_lines.append(header_line)
        for cid, name in enumerate(OUT_CLASS_NAMES):
            gt_n   = int((gt4 == cid).sum())
            pred_n = int((final == cid).sum().item())
            line = f'{name:<14} {gt_n:>10,} {pred_n:>10,}'
            print(line)
            score_lines.append(line)
    else:
        print('  (ingen raw_laz → kan ikke beregne GT-scores)')

    # ── Per-subtile + whole-tile scores (voxel-level, general classes) ──────────
    print('\n=== Scores per subtile + hele tilen ===')
    subtile_ids_gen = [i for i, cid in enumerate(dataset_gen.cloud_ids) if cid.startswith(args.tile)]
    score_lines.append('\n=== Scores per subtile + hele tilen ===')
    class_names_score = ['ground', 'not_ground', 'building']

    # accumulators for whole-tile
    tile_stats = {c: {'tp':0,'fp':0,'fn':0} for c in range(GEN_NC)}

    for si, idx in enumerate(subtile_ids_gen):
        nag_s = dataset_gen[idx]
        nag_s = dataset_gen.on_device_transform(nag_s.to(device))
        with torch.no_grad():
            out_s = model_gen(nag_s)
        pred_s = out_s.voxel_semantic_pred(super_index=nag_s[0].super_index).cpu().numpy()
        gt_s   = nag_s[0].y.argmax(dim=-1).cpu().numpy()
        del out_s, nag_s; torch.cuda.empty_cache()

        valid = gt_s < GEN_NC
        subtile_line = f'\n  --- Subtile {si} ({dataset_gen.cloud_ids[subtile_ids_gen[si]]}) ---'
        print(subtile_line); score_lines.append(subtile_line)

        for c, name in enumerate(class_names_score):
            tp = int(((gt_s[valid]==c) & (pred_s[valid]==c)).sum())
            fp = int(((gt_s[valid]!=c) & (pred_s[valid]==c)).sum())
            fn = int(((gt_s[valid]==c) & (pred_s[valid]!=c)).sum())
            tile_stats[c]['tp'] += tp
            tile_stats[c]['fp'] += fp
            tile_stats[c]['fn'] += fn
            iou  = tp/(tp+fp+fn) if (tp+fp+fn)>0 else float('nan')
            prec = tp/(tp+fp)    if (tp+fp)>0    else float('nan')
            rec  = tp/(tp+fn)    if (tp+fn)>0    else float('nan')
            line = f'    {name:<14}: IoU={100*iou:5.1f}%  Prec={100*prec:5.1f}%  Recall={100*rec:5.1f}%  GT={tp+fn:,}'
            print(line); score_lines.append(line)

    # Whole-tile aggregated scores
    tile_line = '\n  === HELE TILEN (aggregert) ==='
    print(tile_line); score_lines.append(tile_line)
    ious = []
    for c, name in enumerate(class_names_score):
        tp = tile_stats[c]['tp']; fp = tile_stats[c]['fp']; fn = tile_stats[c]['fn']
        iou  = tp/(tp+fp+fn) if (tp+fp+fn)>0 else float('nan')
        prec = tp/(tp+fp)    if (tp+fp)>0    else float('nan')
        rec  = tp/(tp+fn)    if (tp+fn)>0    else float('nan')
        ious.append(iou)
        line = f'    {name:<14}: IoU={100*iou:5.1f}%  Prec={100*prec:5.1f}%  Recall={100*rec:5.1f}%  GT={tp+fn:,}'
        print(line); score_lines.append(line)
    miou_line = f'    {"mIoU":<14}: {100*np.mean(ious):.1f}%'
    print(miou_line); score_lines.append(miou_line)
    tp_all = sum(tile_stats[c]['tp'] for c in range(GEN_NC))
    total  = sum(tile_stats[c]['tp'] + tile_stats[c]['fp'] for c in range(GEN_NC))
    oa_line = f'    {"OA":<14}: {100*tp_all/total:.1f}%' if total > 0 else '    OA: N/A'
    print(oa_line); score_lines.append(oa_line)

    # Save score report
    score_file = out_dir / f'{args.tile}_scores.txt'
    with open(score_file, 'w') as f:
        f.write('\n'.join(score_lines))
    print(f'\nScores saved: {score_file}')

    # ── HTML per subtile ──────────────────────────────────────────────────────
    print('\n=== HTML visualisation (per subtile) ===')
    html_dir = out_dir / args.tile / 'html'
    html_dir.mkdir(parents=True, exist_ok=True)

    for si, idx in enumerate(subtile_ids_gen):
        subtile_name = dataset_gen.cloud_ids[idx]
        nag_vis = dataset_gen[idx]
        nag_vis = dataset_gen.on_device_transform(nag_vis.to(device))
        with torch.no_grad():
            out_vis = model_gen(nag_vis)
        nag_vis[0].semantic_pred = out_vis.voxel_semantic_pred(super_index=nag_vis[0].super_index)
        del out_vis; torch.cuda.empty_cache()

        html_path = str(html_dir / f'subtile_{si:02d}.html')
        nag_vis.show(
            figsize=1600,
            class_names=dataset_gen.class_names,
            class_colors=dataset_gen.class_colors,
            stuff_classes=dataset_gen.stuff_classes,
            num_classes=dataset_gen.num_classes,
            max_points=150_000,
            semantic_pred=True,
            title=f'{args.tile} — subtile {si:02d}/{len(subtile_ids_gen)-1}',
            path=html_path,
            display=False,
        )
        del nag_vis
        print(f'  subtile {si:02d}: {html_path}')

    del model_gen
    torch.cuda.empty_cache()
    print(f'\nAlle HTML lagret i: {html_dir}')


if __name__ == '__main__':
    main()
