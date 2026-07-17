import logging

from src.datamodules.base import BaseDataModule
from src.datasets.bro import BroALS, MiniBroALS

log = logging.getLogger(__name__)

__all__ = ["BroDataModule"]


class BroDataModule(BaseDataModule):
    _DATASET_CLASS = BroALS
    _MINIDATASET_CLASS = MiniBroALS
