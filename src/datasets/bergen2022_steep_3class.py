import os
import os.path as osp
import logging
from typing import Dict, List

import laspy
import numpy as np
import torch
import torch.multiprocessing

from src.data import Data
from src.datasets import BaseDataset
from src.datasets.bergen2022_steep_3class_config import (
    CLASS_COLORS,
    CLASS_NAMES,
    ID2TRAINID,
    TILES,
    BERGEN2022_STEEP_NUM_CLASSES,
)

torch.multiprocessing.set_sharing_strategy("file_system")

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

__all__ = ["Bergen2022Steep3ClassALS", "MiniBergen2022Steep3ClassALS"]


def read_bergen2022_steep_laz(path: str, rgb: bool = True) -> Data:
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


class Bergen2022Steep3ClassALS(BaseDataset):
    """Focused Bergen2022 3-class dataset: 30 steepest tiles, ground /
    not_ground / building."""

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
        return BERGEN2022_STEEP_NUM_CLASSES

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
            f"No auto-download. Run "
            f"scripts/national_model/setup_bergen2022_steep.py first.\n"
            f"{self.raw_file_structure}"
        )

    def read_single_raw_cloud(self, raw_cloud_path: str) -> Data:
        return read_bergen2022_steep_laz(raw_cloud_path, rgb=self.rgb)

    @property
    def raw_file_structure(self) -> str:
        return f"""
{self.root}/
  └── raw/
      └── {{train,val,test}}/
          └── {{tile_id}}.laz   (symlinks to source bergen2022)
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


class MiniBergen2022Steep3ClassALS(Bergen2022Steep3ClassALS):
    _NUM_MINI = 2

    @property
    def all_cloud_ids(self):
        return {k: v[: self._NUM_MINI] for k, v in super().all_cloud_ids.items()}
