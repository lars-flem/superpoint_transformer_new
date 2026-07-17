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
    parser.add_argument('--no_las',      action='store_true', help='Skip saving classified point cloud')
    parser.add_argument('--laz',         action='store_true', help='Save as LAZ instead of LAS (compressed)')
    parser.add_argument('--skip_bridge', action='store_true', help='Skip bridge model (stage 2), use general predictions only')
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
    cfg_gen.datamodule.data_dir = gen_data
    dm_gen = hydra.utils.instantiate(cfg_gen.datamodule)
    dm_gen.prepare_data()
    dm_gen.setup()
    dataset_gen = getattr(dm_gen, f'{args.split}_dataset')

    pos, general_pred = run_inference(dataset_gen, args.tile, model_gen, device)
    print(f'  Points: {len(pos):,}  |  Unique preds: {general_pred.unique().tolist()}')
    # ── Merge predictions ─────────────────────────────────────────────────────
    print('\n=== Merging predictions ===')
    gen_map = {0: OUT_GROUND, 1: OUT_NOT_GROUND, 2: OUT_BUILDING}
    final = torch.full((len(pos),), OUT_IGNORED, dtype=torch.long)
    for src, dst in gen_map.items():
        final[general_pred == src] = dst

    if not args.skip_bridge:
        # ── Stage 2: Bridge model (bruker samme prosesserte subtiles som general) ──
        print('\n=== Stage 2: Bridge model ===')
        _, model_bro = load_model('semantic/bro', args.bridge_ckpt, device)
        _, bridge_pred = run_inference(dataset_gen, args.tile, model_bro, device)
        print(f'  Bridge predictions: {(bridge_pred == 1).sum().item():,}')
        del model_bro
        torch.cuda.empty_cache()

        is_bridge   = bridge_pred == 1
        is_building = general_pred == 2
        mask = is_bridge if args.no_building_mask else is_bridge & ~is_building
        final[mask] = OUT_BRIDGE
        print(f'  Final bridge points: {(final == OUT_BRIDGE).sum().item():,}')
        if not args.no_building_mask:
            print(f'  Bridge predictions blocked by building mask: {(is_bridge & is_building).sum().item():,}')

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
    vox_idx             = None   # set below; reused in per-subtile bridge scoring
    point_bridge_gt_all   = None
    point_bridge_pred_all = None

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

        # Count comparison (full-res GT vs voxel-level pred)
        header_line = f'\n{"Class":<14} {"GT pts":>10} {"Pred vox":>10}'
        print(header_line)
        score_lines.append(header_line)
        for cid, name in enumerate(OUT_CLASS_NAMES):
            gt_n   = int((gt4 == cid).sum())
            pred_n = int((final == cid).sum().item())
            line = f'{name:<14} {gt_n:>10,} {pred_n:>10,}'
            print(line)
            score_lines.append(line)

        # Bridge IoU — propagate voxel predictions to raw points (point-level)
        if not args.skip_bridge:
            from scipy.spatial import cKDTree
            raw_xyz = np.vstack([
                np.asarray(raw_las.x, dtype=np.float64),
                np.asarray(raw_las.y, dtype=np.float64),
                np.asarray(raw_las.z, dtype=np.float64),
            ]).T
            # Build tree on voxel centers, query every raw point → point inherits voxel label
            vox_tree = cKDTree(pos_np)  # pos_np = voxel centers (float64 from line above)
            _, vox_idx = vox_tree.query(raw_xyz, k=1, workers=-1)
            final_arr = final.numpy()
            point_bridge_pred_all = final_arr[vox_idx] == OUT_BRIDGE
            point_bridge_gt_all   = raw_cls == 17
            tp = int((point_bridge_gt_all & point_bridge_pred_all).sum())
            fp = int((~point_bridge_gt_all & point_bridge_pred_all).sum())
            fn = int((point_bridge_gt_all & ~point_bridge_pred_all).sum())
            iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float('nan')
            prec = tp / (tp + fp)       if (tp + fp) > 0       else float('nan')
            rec  = tp / (tp + fn)       if (tp + fn) > 0       else float('nan')
            bridge_line = (f'\nBridge IoU (point-level): {100*iou:.1f}%  '
                           f'Prec={100*prec:.1f}%  Recall={100*rec:.1f}%  '
                           f'GT pts={tp+fn:,}  Pred pts={tp+fp:,}')
            print(bridge_line)
            score_lines.append(bridge_line)
    else:
        print('  (ingen raw_laz → kan ikke beregne GT-scores)')

    # ── Per-subtile + whole-tile scores (voxel-level, general classes) ──────────
    print('\n=== Scores per subtile + hele tilen ===')
    subtile_ids_gen = [i for i, cid in enumerate(dataset_gen.cloud_ids) if cid.startswith(args.tile)]
    score_lines.append('\n=== Scores per subtile + hele tilen ===')
    class_names_score = ['ground', 'not_ground', 'building']

    # accumulators for whole-tile
    tile_stats = {c: {'tp':0,'fp':0,'fn':0} for c in range(GEN_NC)}
    subtile_vox_sizes = []  # track per-subtile voxel counts for HTML bridge overlay

    for si, idx in enumerate(subtile_ids_gen):
        nag_s = dataset_gen[idx]
        nag_s = dataset_gen.on_device_transform(nag_s.to(device))
        with torch.no_grad():
            out_s = model_gen(nag_s)
        pred_s = out_s.voxel_semantic_pred(super_index=nag_s[0].super_index).cpu().numpy()
        gt_s   = nag_s[0].y.argmax(dim=-1).cpu().numpy()
        subtile_offset = sum(subtile_vox_sizes)  # voxel offset before this subtile
        subtile_vox_sizes.append(len(pred_s))
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

        # Per-subtile bridge IoU (point-level, uses vox_idx from bridge IoU block above)
        if vox_idx is not None and not args.skip_bridge:
            n_sub = len(pred_s)
            in_sub = (vox_idx >= subtile_offset) & (vox_idx < subtile_offset + n_sub)
            sub_gt   = point_bridge_gt_all[in_sub]
            sub_pred = point_bridge_pred_all[in_sub]
            btp = int((sub_gt & sub_pred).sum())
            bfp = int((~sub_gt & sub_pred).sum())
            bfn = int((sub_gt & ~sub_pred).sum())
            biou  = btp / (btp + bfp + bfn) if (btp + bfp + bfn) > 0 else float('nan')
            bprec = btp / (btp + bfp)        if (btp + bfp) > 0        else float('nan')
            brec  = btp / (btp + bfn)        if (btp + bfn) > 0        else float('nan')
            bline = f'    {"bridge":<14}: IoU={100*biou:5.1f}%  Prec={100*bprec:5.1f}%  Recall={100*brec:5.1f}%  GT={btp+bfn:,}'
            print(bline); score_lines.append(bline)

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

    subtile_offsets = [sum(subtile_vox_sizes[:i]) for i in range(len(subtile_vox_sizes))]

    for si, idx in enumerate(subtile_ids_gen):
        subtile_name = dataset_gen.cloud_ids[idx]
        nag_vis = dataset_gen[idx]
        nag_vis = dataset_gen.on_device_transform(nag_vis.to(device))
        with torch.no_grad():
            out_vis = model_gen(nag_vis)
        gen_pred_sub = out_vis.voxel_semantic_pred(super_index=nag_vis[0].super_index).cpu()
        del out_vis; torch.cuda.empty_cache()

        # Build 4-class predictions: general model + bridge overlay
        n = subtile_vox_sizes[si]
        offset = subtile_offsets[si]
        final_sub = torch.full((n,), OUT_IGNORED, dtype=torch.long)
        for src, dst in ((0, OUT_GROUND), (1, OUT_NOT_GROUND), (2, OUT_BUILDING)):
            final_sub[gen_pred_sub == src] = dst
        if not args.skip_bridge:
            bridge_sub = bridge_pred[offset:offset + n]
            is_bridge_sub   = bridge_sub == 1
            is_building_sub = gen_pred_sub == 2
            bro_mask = is_bridge_sub if args.no_building_mask else is_bridge_sub & ~is_building_sub
            final_sub[bro_mask] = OUT_BRIDGE
        nag_vis[0].semantic_pred = final_sub.to(device)

        # Fix GT: add bridge column between building and ignore
        # y shape is (N, K): first GEN_NC cols = class one-hots, rest = ignore
        if vox_idx is not None:
            in_sub = (vox_idx >= offset) & (vox_idx < offset + n)
            sub_vox_local = vox_idx[in_sub] - offset
            sub_is_bridge = point_bridge_gt_all[in_sub]
            vox_bridge_gt = np.zeros(n, dtype=bool)
            np.logical_or.at(vox_bridge_gt, sub_vox_local, sub_is_bridge)
            bridge_mask = torch.from_numpy(vox_bridge_gt)
            y = nag_vis[0].y.cpu()
            y3 = y[:, :GEN_NC].clone()
            ignore_col = y[:, GEN_NC:].clone() if y.shape[1] > GEN_NC else torch.zeros(n, 1)
            # Bridge voxels were assigned to not_ground; move them to bridge column
            bridge_col = torch.zeros(n, 1)
            bridge_col[bridge_mask, 0] = 1.0
            y3[bridge_mask, OUT_NOT_GROUND] = 0.0
            nag_vis[0].y = torch.cat([y3, bridge_col, ignore_col], dim=1).to(device)

        vis_class_names  = OUT_CLASS_NAMES + ['ignore']
        vis_class_colors = OUT_CLASS_COLORS + [[50, 50, 50]]
        vis_num_classes  = len(vis_class_names)

        html_path = str(html_dir / f'subtile_{si:02d}.html')
        nag_vis.show(
            figsize=1600,
            class_names=vis_class_names,
            class_colors=vis_class_colors,
            stuff_classes=dataset_gen.stuff_classes,
            num_classes=vis_num_classes,
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
