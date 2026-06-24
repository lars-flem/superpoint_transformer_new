import logging

from src.datamodules.base import BaseDataModule
from src.datasets.bergen2022_steep_3class import (
    Bergen2022Steep3ClassALS,
    MiniBergen2022Steep3ClassALS,
)

log = logging.getLogger(__name__)

__all__ = ["Bergen2022Steep3ClassDataModule"]


class Bergen2022Steep3ClassDataModule(BaseDataModule):
    _DATASET_CLASS = Bergen2022Steep3ClassALS
    _MINIDATASET_CLASS = MiniBergen2022Steep3ClassALS
