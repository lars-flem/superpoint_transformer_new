import os
import os.path as osp
import logging
from typing import Dict, List

import numpy as np
import torch
import torch.multiprocessing

from src.data import Data
from src.datasets import BaseDataset
from src.datasets.norway_combined_binary_config import (
    CLASS_COLORS,
    CLASS_NAMES,
    ID2TRAINID,
    NORWAY_COMBINED_BINARY_NUM_CLASSES,
    REGIONS,
    TILES,
    TILES_MINI,
)
from src.datasets.norway_combined_3class import read_norway_combined_laz

torch.multiprocessing.set_sharing_strategy("file_system")

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

__all__ = [
    "NorwayCombinedBinaryALS",
    "MiniNorwayCombinedBinaryALS",
]


# The binary read function reuses the same LAZ loader as the 3-class variant
# but applies the binary ID2TRAINID. We re-define a thin wrapper rather than
# importing read_norway_combined_laz directly because that function bakes in
# the 3-class ID2TRAINID at import time.
def _read_binary(path: str, rgb: bool = True) -> Data:
    import laspy
    las = laspy.read(path)
    data = Data()

    pos = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    pos = torch.from_numpy(pos)
    offset = pos[0]
    data.pos = pos - offset
    data.pos_offset = offset

    cls = np.asarray(las.classification, dtype=np.int64)
    y = ID2TRAINID[cls]
    data.y = torch.from_numpy(y).long()

    if rgb:
        try:
            has_rgb = all(
                dim in las.point_format.dimension_names
                for dim in ["red", "green", "blue"]
            )
            if has_rgb:
                red = np.asarray(las.red, dtype=np.uint16)
                green = np.asarray(las.green, dtype=np.uint16)
                blue = np.asarray(las.blue, dtype=np.uint16)
                rgb_data = np.vstack([red, green, blue]).T.astype(np.float32)
                rgb_tensor = torch.from_numpy(rgb_data)
                data.rgb = (rgb_tensor / 65535.0).clamp(min=0, max=1)
        except Exception as exc:
            log.warning(f"Failed to load RGB from {path}: {exc}")

    if "intensity" in las.point_format.dimension_names:
        inten = np.asarray(las.intensity, dtype=np.float32)
        denom = float(inten.max()) if inten.max() > 0 else 1.0
        data.intensity = torch.from_numpy(inten / denom)

    return data


class NorwayCombinedBinaryALS(BaseDataset):
    """Combined Viken + Oslo + Bergen2020 + Bergen2022 ALS dataset, binary
    (ground / not_ground). Tile ids are region-prefixed.
    """

    def __init__(self, *args, rgb: bool = True, **kwargs):
        self.rgb = rgb
        super().__init__(*args, **kwargs)

    @property
    def data_subdir_name(self) -> str:
        return ""

    @property
    def class_names(self) -> List[str]:
        return CLASS_NAMES

    @property
    def num_classes(self) -> int:
        return NORWAY_COMBINED_BINARY_NUM_CLASSES

    @property
    def stuff_classes(self) -> List[int]:
        return list(range(self.num_classes))

    @property
    def class_colors(self):
        return CLASS_COLORS

    @property
    def all_base_cloud_ids(self) -> Dict[str, List[str]]:
        return TILES

    def download_dataset(self) -> None:
        raise RuntimeError(
            f"No auto-download. Run scripts/setup_norway_combined.py first.\n"
            f"Expected layout:\n{self.raw_file_structure}"
        )

    def read_single_raw_cloud(self, raw_cloud_path: str) -> Data:
        return _read_binary(raw_cloud_path, rgb=self.rgb)

    @property
    def raw_file_structure(self) -> str:
        return f"""
{self.root}/
  └── raw/
      └── {{train,val,test}}/
          └── {{region}}_{{tile_id}}.laz   (symlinks created by setup script)
"""

    def id_to_relative_raw_path(self, cloud_id: str) -> str:
        base_id = self.id_to_base_id(cloud_id)
        for stage in ("train", "val", "test"):
            if base_id in self.all_base_cloud_ids[stage]:
                return osp.join(stage, base_id + ".laz")
        raise ValueError(f"Unknown tile id '{cloud_id}' (base_id='{base_id}')")

    def processed_to_raw_path(self, processed_path: str) -> str:
        _, _, cloud_id = osp.splitext(processed_path)[0].split(osp.sep)[-3:]
        base_cloud_id = self.id_to_base_id(cloud_id)
        for stage in ("train", "val", "test"):
            if base_cloud_id in self.all_base_cloud_ids[stage]:
                return osp.join(self.raw_dir, stage, base_cloud_id + ".laz")
        raise ValueError(
            f"Unknown cloud id '{base_cloud_id}' (not in train/val/test lists)"
        )


class MiniNorwayCombinedBinaryALS(NorwayCombinedBinaryALS):
    """Mini variant for development."""

    @property
    def all_base_cloud_ids(self) -> Dict[str, List[str]]:
        return TILES_MINI
