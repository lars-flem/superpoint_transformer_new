"""
Generate h5 files for all tiles in national_data by running preprocessing only.
No inference — just triggers datamodule.setup('test') to cache h5 to disk.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path('/cluster/home/larshfle/superpoint_transformer_new')
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import hydra
import torch
import src.datasets.norway_combined_3class_config as gen_cfg
import src.datasets.norway_combined_3class as gen_mod

from src.utils import init_config
from src.transforms import *
from src.data import *

DATA_DIR = Path('/cluster/home/larshfle/datasets/national_data')

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}')

# Only preprocess the two new Arendal tiles
tile_ids = [
    'arendal_island_32-1-497-99-41',
    'arendal_coast_32-1-497-98-25',
]
print(f'\nTiles to preprocess ({len(tile_ids)}):')
for t in tile_ids:
    print(f'  {t}')

gen_cfg.TILES = gen_mod.TILES = {'train': [], 'val': [], 'test': tile_ids}

cfg = init_config(overrides=[
    'experiment=semantic/norway_combined_3class',
    'datamodule=semantic/norway_combined_3class',
    f'paths.data_dir={DATA_DIR}',
])

print('\nStarting preprocessing...')
datamodule = hydra.utils.instantiate(cfg.datamodule)
datamodule.setup('test')

print('\nDone! H5 files saved to:')
print(f'  {DATA_DIR}/processed/')
