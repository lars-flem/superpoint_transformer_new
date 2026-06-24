"""Export full-tile and cropped-region HTML visualizations for the
bergen2022_steep_3class run.

Mirrors `scripts/viken/visualize_gjerdrum_viken.py`: loads the checkpoint,
runs inference on one test sub-tile, and writes two HTML files —
one full-tile and one 50 m crop — into
`visualizations/bergen2022_steep/`.

Run on a GPU node (preprocessing/partition needs CUDA). The fastest path
is `datamodule.prepare_only_test=true`, which skips train/val dataset
construction.
"""

import os
import sys
import gc

import hydra
import torch

REPO = "/cluster/home/jakobep/superpoint_transformer_new"
sys.path.insert(0, REPO)

from src.utils import init_config  # noqa: E402

OUT_DIR = os.path.join(REPO, "visualizations", "bergen2022_steep")
os.makedirs(OUT_DIR, exist_ok=True)

CROP_RADIUS = 50  # m

# Each entry: (name, run_id, experiment, datamodule, data_dir, xy_tiling, test_tile)
RUNS = [
    ("bergen2022_steep_3class",
     "2026-05-27_08-57-07",
     "semantic/bergen2022_steep_3class",
     "semantic/bergen2022_steep_3class",
     "/cluster/home/jakobep/datasets/bergen2022_steep_3class",
     3,
     "32-1-467-145-40__TILE_2-2_OF_3-3"),
]


def visualize_run(name, ckpt_run, experiment, datamodule, data_dir,
                  xy_tiling, test_tile, device):
    ckpt_path = os.path.join(
        REPO, "logs", "train", "runs", ckpt_run, "checkpoints", "last.ckpt"
    )
    assert os.path.exists(ckpt_path), f"missing checkpoint: {ckpt_path}"

    print(f"\n=== {name} ===")
    print(f"  ckpt: {ckpt_path}")
    print(f"  data_dir: {data_dir}")
    print(f"  xy_tiling: {xy_tiling}")
    print(f"  test_tile: {test_tile}")

    cfg = init_config(overrides=[
        f"experiment={experiment}",
        f"datamodule={datamodule}",
        f"ckpt_path={ckpt_path}",
        f"datamodule.data_dir={data_dir}",
        f"datamodule.xy_tiling={xy_tiling}",
        "datamodule.mini=false",
        "datamodule.load_full_res_idx=true",
        # Skip train/val construction — much faster setup
        "datamodule.prepare_only_test=true",
    ])

    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.prepare_data()
    dm.setup()
    dataset = dm.test_dataset

    tile_idx = next(
        i for i, cid in enumerate(dataset.cloud_ids) if cid.startswith(test_tile)
    )
    print(f"  matched tile: {dataset.cloud_ids[tile_idx]} (idx {tile_idx})")

    model = hydra.utils.instantiate(cfg.model)
    load_kwargs = {}
    pretrained_cnn = cfg.datamodule.get("pretrained_cnn_ckpt_path", None)
    if pretrained_cnn is not None:
        load_kwargs["pretrained_cnn_ckpt_path"] = pretrained_cnn
    model = model._load_from_checkpoint(cfg.ckpt_path, **load_kwargs)
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
        title=f"{name} — full sub-tile ({test_tile})",
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
        title=f"{name} — {CROP_RADIUS} m crop ({test_tile})",
        path=crop_path,
        display=False,
        **common_kwargs,
    )
    print(f"  wrote: {crop_path}")

    del model, dm, dataset, nag, output
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
