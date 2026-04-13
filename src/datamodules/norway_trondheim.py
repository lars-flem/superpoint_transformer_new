import logging
from src.datamodules.base import BaseDataModule
from src.datasets.norway_trondheim2022 import NorwayTrondheimALS, MiniNorwayTrondheimALS


log = logging.getLogger(__name__)
    

__all__ = ["NorwayTrondheimDataModule"]


class NorwayTrondheimDataModule(BaseDataModule):
    _DATASET_CLASS = NorwayTrondheimALS
    _MINIDATASET_CLASS = MiniNorwayTrondheimALS
