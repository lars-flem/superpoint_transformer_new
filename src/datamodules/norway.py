import logging
from src.datamodules.base import BaseDataModule
from src.datasets.norway_bergen2022 import NorwayALS, MiniNorwayALS


log = logging.getLogger(__name__)
    

__all__ = ["NorwayDataModule"]


class NorwayDataModule(BaseDataModule):
    _DATASET_CLASS = NorwayALS
    _MINIDATASET_CLASS = MiniNorwayALS
