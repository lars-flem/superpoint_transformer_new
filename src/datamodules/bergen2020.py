import logging
from src.datamodules.base import BaseDataModule
from src.datasets.bergen2020 import (
    Bergen2020ALS, MiniBergen2020ALS,
    Bergen2020BinaryALS, MiniBergen2020BinaryALS,
)

log = logging.getLogger(__name__)

__all__ = ['Bergen2020DataModule', 'Bergen2020BinaryDataModule']


class Bergen2020DataModule(BaseDataModule):
    _DATASET_CLASS = Bergen2020ALS
    _MINIDATASET_CLASS = MiniBergen2020ALS


class Bergen2020BinaryDataModule(BaseDataModule):
    _DATASET_CLASS = Bergen2020BinaryALS
    _MINIDATASET_CLASS = MiniBergen2020BinaryALS
