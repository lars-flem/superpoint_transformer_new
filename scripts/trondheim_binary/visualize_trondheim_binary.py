"""Export full-tile and cropped-region HTML visualizations for the Trondheim
point-density experiments (binary + multiclass × {5, 10, 20, 30, 30down}
pkt/m^2).

Mirrors the export logic from notebooks/viken2022_visualization.ipynb:
loads each trained model, runs inference on one test sub-tile, and writes
two HTML files per run — one full-tile, one cropped region — into
`visualizations/trondheim_density/`.
"""

import os
import sys
import gc
import torch
import hydra

REPO = "/cluster/home/larshfle/superpoint_transformer_new"
sys.path.insert(0, REPO)

from src.utils import init_config  # noqa: E402

OUT_DIR = os.path.join(REPO, "visualizations", "trondheim_density")
os.makedirs(OUT_DIR, exist_ok=True)

# The test sub-tile to load. With xy_tiling=2 (most runs) and xy_tiling=4
# (trondheim5 binary) the same base tile is split differently, so we match
# the cloud_id by `startswith`.
TEST_TILE = "32-1-510-215-60"

# Cropped-region radius (m).
CROP_RADIUS = 50

# Each entry: (run_id, ckpt_dir, experiment, data_dir, xy_tiling).
RUNS = [
    ("trondheim2022_binary",
     "/cluster/home/larshfle/superpoint_transformer_new/logs/train/multiruns/2026-03-21_22-38-52/0/checkpoints/epoch_399.ckpt",
     "semantic/norway_trondheim2022",
     "/cluster/home/larshfle/datasets/trondheim2022_30pkt",
     3),
]


def visualize_run(name, ckpt_path, experiment, data_dir, xy_tiling, device):
    assert os.path.exists(ckpt_path), f"missing checkpoint: {ckpt_path}"

    print(f"\n=== {name} ===")
    print(f"  ckpt: {ckpt_path}")
    print(f"  data_dir: {data_dir}")
    print(f"  xy_tiling: {xy_tiling}")

    cfg = init_config(overrides=[
        f"experiment={experiment}",
        f"ckpt_path={ckpt_path}",
        f"datamodule.data_dir={data_dir}",
        f"datamodule.xy_tiling={xy_tiling}",
        "datamodule.mini=false",
        "datamodule.load_full_res_idx=true",
    ])

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup()
    dataset = datamodule.test_dataset

    tile_idx = next(
        i for i, cid in enumerate(dataset.cloud_ids) if cid.startswith(TEST_TILE)
    )
    print(f"  tile: {dataset.cloud_ids[tile_idx]} (idx {tile_idx})")

    model = hydra.utils.instantiate(cfg.model)
    load_kwargs = {}
    pretrained_cnn = cfg.datamodule.get("pretrained_cnn_ckpt_path", None)
    if pretrained_cnn is not None:
        load_kwargs["pretrained_cnn_ckpt_path"] = pretrained_cnn
    model = model._load_from_checkpoint(ckpt_path, **load_kwargs)
    model = model.eval().to(device)
    model.net.store_features = True

    nag = dataset[tile_idx]
    nag = dataset.on_device_transform(nag.to(device))

    print(f"  level-0 points: {nag[0].pos.shape[0]:,}")

    with torch.no_grad():
        output = model(nag)

    nag[0].semantic_pred = output.voxel_semantic_pred(super_index=nag[0].super_index)
    if hasattr(output, "voxel_panoptic_pred"):
        try:
            _, _, vox_obj_pred = output.voxel_panoptic_pred(
                super_index=nag[0].super_index
            )
            nag[0].obj_pred = vox_obj_pred
        except Exception as exc:
            print(f"  (panoptic pred skipped: {exc})")

    common_kwargs = dict(
        class_names=dataset.class_names,
        class_colors=dataset.class_colors,
        stuff_classes=dataset.stuff_classes,
        num_classes=dataset.num_classes,
        max_points=100_000,
    )

    full_path = os.path.join(OUT_DIR, f"{name}_full.html")
    nag.show(
        figsize=1600,
        title=f"{name} — full sub-tile ({TEST_TILE})",
        path=full_path,
        display=False,
        **common_kwargs,
    )
    print(f"  wrote: {full_path}")

    center = nag[0].pos.mean(dim=0).view(1, -1)
    crop_path = os.path.join(OUT_DIR, f"{name}_crop.html")
    nag.show(
        figsize=1600,
        radius=CROP_RADIUS,
        center=center,
        title=f"{name} — {CROP_RADIUS} m crop ({TEST_TILE})",
        path=crop_path,
        display=False,
        **common_kwargs,
    )
    print(f"  wrote: {crop_path}")

    del model, datamodule, dataset, nag, output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for entry in RUNS:
        try:
            visualize_run(*entry, device=device)
        except Exception as exc:
            print(f"!! {entry[0]} failed: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
