import logging

from src.datamodules.base import BaseDataModule
from src.datasets.norway_combined_binary import (
    MiniNorwayCombinedBinaryALS,
    NorwayCombinedBinaryALS,
)

log = logging.getLogger(__name__)

__all__ = ["NorwayCombinedBinaryDataModule"]


class NorwayCombinedBinaryDataModule(BaseDataModule):
    _DATASET_CLASS = NorwayCombinedBinaryALS
    _MINIDATASET_CLASS = MiniNorwayCombinedBinaryALS
