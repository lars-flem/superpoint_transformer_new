import logging

from src.datamodules.base import BaseDataModule
from src.datasets.viken2022 import MiniViken2022ALS, Viken2022ALS

log = logging.getLogger(__name__)

__all__ = ["Viken2022DataModule"]


class Viken2022DataModule(BaseDataModule):
    _DATASET_CLASS = Viken2022ALS
    _MINIDATASET_CLASS = MiniViken2022ALS
