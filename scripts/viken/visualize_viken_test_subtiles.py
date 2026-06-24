"""Visualize all 4 sub-tiles of the viken2022 test tile 32-1-513-134-62 with
the trained multiclass model, and export a LAZ per sub-tile holding the
predicted class labels.

Adapted from `visualize_gjerdrum_viken.py`: loads the viken2022 checkpoint,
runs inference on every sub-tile whose cloud_id starts with the test tile,
and writes a full-tile HTML, a 50 m crop HTML, and a predicted-label LAZ into
`visualizations/viken_test_subtiles/`.
"""

import os
import sys
import gc
import numpy as np
import torch
import hydra
import laspy

REPO = "/cluster/home/jakobep/superpoint_transformer_new"
sys.path.insert(0, REPO)

from src.utils import init_config  # noqa: E402

OUT_DIR = os.path.join(REPO, "visualizations", "viken_test_subtiles")
os.makedirs(OUT_DIR, exist_ok=True)

CROP_RADIUS = 50

CKPT_RUN = "2026-03-31_23-50-44"
EXPERIMENT = "semantic/viken2022"
DATAMODULE = "semantic/viken2022"
DATA_DIR = "/cluster/home/jakobep/datasets/viken2022"
XY_TILING = 2
TEST_TILE = "32-1-513-134-62"


def export_laz(path, pos_xyz, labels):
    """Write XYZ + predicted class label to a LAZ file."""
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = pos_xyz.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = pos_xyz[:, 0]
    las.y = pos_xyz[:, 1]
    las.z = pos_xyz[:, 2]
    las.classification = labels.astype(np.uint8)
    las.write(path)
    print(f"  wrote: {path}  ({len(labels):,} pts)")


def visualize_subtile(dataset, model, tile_idx, device):
    name = dataset.cloud_ids[tile_idx]
    print(f"\n=== {name} (idx {tile_idx}) ===")

    nag = dataset[tile_idx]
    nag = dataset.on_device_transform(nag.to(device))
    print(f"  level-0 points: {nag[0].pos.shape[0]:,}")

    with torch.no_grad():
        output = model(nag)

    nag[0].semantic_pred = output.voxel_semantic_pred(
        super_index=nag[0].super_index
    )

    common_kwargs = dict(
        class_names=dataset.class_names,
        class_colors=dataset.class_colors,
        stuff_classes=dataset.stuff_classes,
        num_classes=dataset.num_classes,
        max_points=100_000,
    )

    safe = name.replace("/", "_")

    full_path = os.path.join(OUT_DIR, f"{safe}_full.html")
    nag.show(
        figsize=1600,
        title=f"viken2022 mc — {name}",
        path=full_path,
        display=False,
        **common_kwargs,
    )
    print(f"  wrote: {full_path}")

    center = nag[0].pos.mean(dim=0).view(1, -1)
    crop_path = os.path.join(OUT_DIR, f"{safe}_crop.html")
    nag.show(
        figsize=1600,
        radius=CROP_RADIUS,
        center=center,
        title=f"viken2022 mc — {name} {CROP_RADIUS} m crop",
        path=crop_path,
        display=False,
        **common_kwargs,
    )
    print(f"  wrote: {crop_path}")

    # Predicted-label LAZ at voxel (level-0) resolution. Absolute coords are
    # recovered by adding back the per-tile position offset.
    pos = (nag[0].pos + nag[0].pos_offset).cpu().numpy().astype(np.float64)
    labels = nag[0].semantic_pred.cpu().numpy()
    export_laz(os.path.join(OUT_DIR, f"{safe}_pred.laz"), pos, labels)

    del nag, output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt_path = os.path.join(
        REPO, "logs", "train", "runs", CKPT_RUN, "checkpoints", "last.ckpt"
    )
    assert os.path.exists(ckpt_path), f"missing checkpoint: {ckpt_path}"
    print(f"ckpt: {ckpt_path}")

    cfg = init_config(overrides=[
        f"experiment={EXPERIMENT}",
        f"datamodule={DATAMODULE}",
        f"ckpt_path={ckpt_path}",
        f"datamodule.data_dir={DATA_DIR}",
        f"datamodule.xy_tiling={XY_TILING}",
        "datamodule.mini=false",
        "datamodule.load_full_res_idx=true",
    ])

    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.prepare_data()
    dm.setup()
    dataset = dm.test_dataset

    model = hydra.utils.instantiate(cfg.model)
    load_kwargs = {}
    pretrained_cnn = cfg.datamodule.get("pretrained_cnn_ckpt_path", None)
    if pretrained_cnn is not None:
        load_kwargs["pretrained_cnn_ckpt_path"] = pretrained_cnn
    model = model._load_from_checkpoint(cfg.ckpt_path, **load_kwargs)
    model = model.eval().to(device)
    model.net.store_features = True

    tile_indices = [
        i for i, cid in enumerate(dataset.cloud_ids)
        if cid.startswith(TEST_TILE)
    ]
    print(f"Found {len(tile_indices)} sub-tiles for {TEST_TILE}: "
          f"{[dataset.cloud_ids[i] for i in tile_indices]}")

    for tile_idx in tile_indices:
        try:
            visualize_subtile(dataset, model, tile_idx, device)
        except Exception as exc:
            print(f"!! sub-tile idx {tile_idx} failed: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
