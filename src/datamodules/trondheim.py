import logging

from src.datamodules.base import BaseDataModule
from src.datasets.trondheim import MiniTrondheimALS, TrondheimALS

log = logging.getLogger(__name__)

__all__ = ["TrondheimDataModule"]


class TrondheimDataModule(BaseDataModule):
    _DATASET_CLASS = TrondheimALS
    _MINIDATASET_CLASS = MiniTrondheimALS
