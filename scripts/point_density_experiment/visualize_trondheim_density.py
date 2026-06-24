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

REPO = "/cluster/home/jakobep/superpoint_transformer_new"
sys.path.insert(0, REPO)

from src.utils import init_config  # noqa: E402

OUT_DIR = os.path.join(REPO, "visualizations", "trondheim_density")
os.makedirs(OUT_DIR, exist_ok=True)

# The test sub-tile to load. With xy_tiling=2 (most runs) and xy_tiling=4
# (trondheim5 binary) the same base tile is split differently, so we match
# the cloud_id by `startswith`.
TEST_TILE = "32-1-511-215-24"

# Cropped-region radius (m).
CROP_RADIUS = 50

# Each entry: (run_id, ckpt_dir, experiment, data_dir, xy_tiling).
RUNS = [
    # xy_tiling=2 binary
    ("trondheim5_binary",             "2026-04-08_13-56-26", "semantic/trondheim5",                  "/cluster/home/jakobep/datasets/trondheim5",         2),
    ("trondheim10_binary",            "2026-04-26_20-17-56", "semantic/trondheim10",                 "/cluster/home/jakobep/datasets/trondheim10",        2),
    ("trondheim20_binary",            "2026-04-26_22-41-58", "semantic/trondheim20",                 "/cluster/home/jakobep/datasets/trondheim20",        2),
    ("trondheim30_binary",            "2026-04-27_00-48-05", "semantic/trondheim30",                 "/cluster/home/jakobep/datasets/trondheim30",        2),
    ("trondheim30down_binary",        "2026-05-06_07-39-34", "semantic/trondheim30down",             "/cluster/home/jakobep/datasets/trondheim30down",    2),
    # xy_tiling=2 multiclass
    # NOTE: trondheim5/10/20/30_multiclass xt=2 are skipped — their test
    # pre_transform cache no longer matches what training used (the multiclass
    # YAML was edited after April training, changing the hash key). Re-running
    # would produce predictions that don't match the original test metrics.
    # Keep their existing Apr-29 crops as-is. 30down_multiclass has intact
    # cache from May and is safe to re-render.
    # ("trondheim5_multiclass",         "2026-04-13_18-40-27", "semantic/trondheim5_multiclass",       "/cluster/home/jakobep/datasets/trondheim5_mc",      2),
    # ("trondheim10_multiclass",        "2026-04-26_16-02-39", "semantic/trondheim10_multiclass",      "/cluster/home/jakobep/datasets/trondheim10_mc",     2),
    # ("trondheim20_multiclass",        "2026-04-22_22-03-38", "semantic/trondheim20_multiclass",      "/cluster/home/jakobep/datasets/trondheim20_mc",     2),
    # ("trondheim30_multiclass",        "2026-04-26_17-00-33", "semantic/trondheim30_multiclass",      "/cluster/home/jakobep/datasets/trondheim30_mc",     2),
    ("trondheim30down_multiclass",    "2026-05-06_10-02-51", "semantic/trondheim30down_multiclass",  "/cluster/home/jakobep/datasets/trondheim30down_mc", 2),
    # xy_tiling=3 binary
    ("trondheim5_binary_xt3",         "2026-05-11_03-11-32", "semantic/trondheim5",                  "/cluster/home/jakobep/datasets/trondheim5",         3),
    ("trondheim10_binary_xt3",        "2026-05-11_06-29-06", "semantic/trondheim10",                 "/cluster/home/jakobep/datasets/trondheim10",        3),
    ("trondheim20_binary_xt3",        "2026-05-07_02-02-13", "semantic/trondheim20",                 "/cluster/home/jakobep/datasets/trondheim20",        3),
    ("trondheim30_binary_xt3",        "2026-05-07_05-41-48", "semantic/trondheim30",                 "/cluster/home/jakobep/datasets/trondheim30",        3),
    ("trondheim30down_binary_xt3",    "2026-05-07_11-07-10", "semantic/trondheim30down",             "/cluster/home/jakobep/datasets/trondheim30down",    3),
    # xy_tiling=3 multiclass
    ("trondheim5_multiclass_xt3",     "2026-05-11_09-01-37", "semantic/trondheim5_multiclass",       "/cluster/home/jakobep/datasets/trondheim5_mc",      3),
    ("trondheim10_multiclass_xt3",    "2026-05-11_20-53-25", "semantic/trondheim10_multiclass",      "/cluster/home/jakobep/datasets/trondheim10_mc",     3),
    ("trondheim20_multiclass_xt3",    "2026-05-07_15-12-47", "semantic/trondheim20_multiclass",      "/cluster/home/jakobep/datasets/trondheim20_mc",     3),
    ("trondheim30_multiclass_xt3",    "2026-05-07_18-44-02", "semantic/trondheim30_multiclass",      "/cluster/home/jakobep/datasets/trondheim30_mc",     3),
    ("trondheim30down_multiclass_xt3","2026-05-08_00-20-01", "semantic/trondheim30down_multiclass",  "/cluster/home/jakobep/datasets/trondheim30down_mc", 3),
]


def visualize_run(name, ckpt_run, experiment, data_dir, xy_tiling, device):
    ckpt_path = os.path.join(
        REPO, "logs", "train", "runs", ckpt_run, "checkpoints", "last.ckpt"
    )
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
        # Skip train/val dataset construction in both prepare_data and setup.
        # We only need the test dataset for visualization.
        "datamodule.prepare_only_test=true",
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
    if os.path.exists(full_path):
        print(f"  skipping full (already exists): {full_path}")
    else:
        nag.show(
            figsize=1600,
            title=f"{name} — full sub-tile ({TEST_TILE})",
            path=full_path,
            display=False,
            **common_kwargs,
        )
        print(f"  wrote: {full_path}")

    # Fixed local-frame crop center (XY chosen to match the patch we want
    # across all density variants). Z uses the per-run mean so the crop sits
    # at the right elevation regardless of the local-frame Z offset.
    pos = nag[0].pos
    mean_z = pos[:, 2].mean()
    center = torch.tensor(
        [[-90.0, 132.0, float(mean_z)]],
        dtype=pos.dtype,
        device=pos.device,
    )
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
