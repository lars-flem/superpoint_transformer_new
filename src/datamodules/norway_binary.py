import logging
from src.datamodules.base import BaseDataModule
from src.datasets.norway_binary import NorwayBinaryALS, MiniNorwayBinaryALS


log = logging.getLogger(__name__)
    

__all__ = ["NorwayBinaryDataModule"]


class NorwayBinaryDataModule(BaseDataModule):
    _DATASET_CLASS = NorwayBinaryALS
    _MINIDATASET_CLASS = MiniNorwayBinaryALS
