import logging

from src.datamodules.base import BaseDataModule
from src.datasets.trondheim_multiclass import MiniTrondheimMulticlassALS, TrondheimMulticlassALS

log = logging.getLogger(__name__)

__all__ = ["TrondheimMulticlassDataModule"]


class TrondheimMulticlassDataModule(BaseDataModule):
    _DATASET_CLASS = TrondheimMulticlassALS
    _MINIDATASET_CLASS = MiniTrondheimMulticlassALS
