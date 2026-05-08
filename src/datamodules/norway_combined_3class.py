import logging

from src.datamodules.base import BaseDataModule
from src.datasets.norway_combined_3class import (
    MiniNorwayCombined3ClassALS,
    NorwayCombined3ClassALS,
)

log = logging.getLogger(__name__)

__all__ = ["NorwayCombined3ClassDataModule"]


class NorwayCombined3ClassDataModule(BaseDataModule):
    _DATASET_CLASS = NorwayCombined3ClassALS
    _MINIDATASET_CLASS = MiniNorwayCombined3ClassALS
