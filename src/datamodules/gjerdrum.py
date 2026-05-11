import logging

from src.datamodules.base import BaseDataModule
from src.datasets.gjerdrum import GjerdrumALS, MiniGjerdrumALS

log = logging.getLogger(__name__)

__all__ = ["GjerdrumDataModule"]


class GjerdrumDataModule(BaseDataModule):
    _DATASET_CLASS = GjerdrumALS
    _MINIDATASET_CLASS = MiniGjerdrumALS
