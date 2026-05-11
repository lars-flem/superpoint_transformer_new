import logging
import os
import os.path as osp
from typing import Dict, List

import laspy
import numpy as np
import torch
import torch.multiprocessing

from src.data import Data
from src.datasets import BaseDataset
import src.datasets.bergen2020_config as cfg3
import src.datasets.bergen2020_binary_config as cfgbin

torch.multiprocessing.set_sharing_strategy('file_system')

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

__all__ = ['Bergen2020ALS', 'MiniBergen2020ALS', 'Bergen2020BinaryALS', 'MiniBergen2020BinaryALS']


def read_bergen2020_laz(path: str, rgb: bool = False) -> Data:
    las = laspy.read(path)
    data = Data()

    pos = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    pos = torch.from_numpy(pos)
    offset = pos[0]
    data.pos = pos - offset
    data.pos_offset = offset

    # Classification mapping applied in subclass via _id2trainid
    cls = np.asarray(las.classification, dtype=np.int64)
    data._raw_cls = cls  # stored temporarily; subclass sets data.y

    if rgb:
        try:
            has_rgb = all(d in las.point_format.dimension_names for d in ['red', 'green', 'blue'])
            if has_rgb:
                r = np.asarray(las.red,   dtype=np.float32)
                g = np.asarray(las.green, dtype=np.float32)
                b = np.asarray(las.blue,  dtype=np.float32)
                data.rgb = torch.from_numpy(
                    np.vstack([r, g, b]).T / 65535.0).clamp(0, 1)
            else:
                log.warning(f'No RGB in {path}')
        except Exception as e:
            log.warning(f'RGB load failed for {path}: {e}')

    if 'intensity' in las.point_format.dimension_names:
        inten = np.asarray(las.intensity, dtype=np.float32)
        denom = float(inten.max()) if inten.max() > 0 else 1.0
        data.intensity = torch.from_numpy(inten / denom)

    return data


class _Bergen2020Base(BaseDataset):
    _CFG = None  # override in subclass

    def __init__(self, *args, rgb: bool = True, **kwargs):
        self.rgb = rgb
        super().__init__(*args, **kwargs)

    @property
    def data_subdir_name(self) -> str:
        return ''

    @property
    def class_names(self) -> List[str]:
        return self._CFG.CLASS_NAMES

    @property
    def num_classes(self) -> int:
        return self._CFG.BERGEN2020_NUM_CLASSES if hasattr(self._CFG, 'BERGEN2020_NUM_CLASSES') \
            else self._CFG.BERGEN2020_BINARY_NUM_CLASSES

    @property
    def stuff_classes(self) -> List[int]:
        return list(range(self.num_classes))

    @property
    def class_colors(self):
        return self._CFG.CLASS_COLORS

    @property
    def all_base_cloud_ids(self) -> Dict[str, List[str]]:
        return self._CFG.TILES

    def download_dataset(self) -> None:
        raise RuntimeError(
            f'No auto-download. Run scripts/setup_bergen2020.sh first.\n{self.raw_file_structure}')

    def read_single_raw_cloud(self, raw_cloud_path: str) -> Data:
        data = read_bergen2020_laz(raw_cloud_path, rgb=self.rgb)
        data.y = torch.from_numpy(self._CFG.ID2TRAINID[data._raw_cls]).long()
        del data._raw_cls
        return data

    @property
    def raw_file_structure(self) -> str:
        return f"""
{self.root}/
  raw/
    train/ val/ test/
      {{tile_id}}.laz
"""

    def id_to_relative_raw_path(self, id: str) -> str:
        base_id = self.id_to_base_id(id)
        for stage in ('train', 'val', 'test'):
            if base_id in self._CFG.TILES[stage]:
                return osp.join(stage, base_id + '.laz')
        raise ValueError(f'Unknown tile: {id}')

    def processed_to_raw_path(self, processed_path: str) -> str:
        _, _, cloud_id = osp.splitext(processed_path)[0].split(osp.sep)[-3:]
        base_id = self.id_to_base_id(cloud_id)
        for stage in ('train', 'val', 'test'):
            if base_id in self._CFG.TILES[stage]:
                return osp.join(self.raw_dir, stage, base_id + '.laz')
        raise ValueError(f'Unknown tile: {cloud_id}')


class Bergen2020ALS(_Bergen2020Base):
    _CFG = cfg3


class MiniBergen2020ALS(Bergen2020ALS):
    @property
    def all_base_cloud_ids(self):
        return {k: v[:2] for k, v in cfg3.TILES.items()}


class Bergen2020BinaryALS(_Bergen2020Base):
    _CFG = cfgbin


class MiniBergen2020BinaryALS(Bergen2020BinaryALS):
    @property
    def all_base_cloud_ids(self):
        return {k: v[:2] for k, v in cfgbin.TILES.items()}
