"""
Runner for national 3-class model inference on new tiles.
Edit TILES_TO_RUN to add/remove tiles.
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path('/cluster/home/larshfle/superpoint_transformer_new')
PYTHON       = Path('/cluster/home/larshfle/.conda/envs/spt/bin/python')

GENERAL_CKPT = PROJECT_ROOT / 'logs/train/runs/2026-05-20_16-26-44/checkpoints/last.ckpt'
GENERAL_DATA = '/cluster/home/larshfle/datasets/norway_combined_3class'
BRIDGE_CKPT  = PROJECT_ROOT / 'logs/train/runs/2026-05-06_19-01-24/checkpoints/epoch_399.ckpt'
BRIDGE_DATA  = '/cluster/home/larshfle/datasets/bro'
OUT_DIR      = PROJECT_ROOT / 'output_pipeline'

OSLO_RAW = Path('/cluster/home/larshfle/datasets/norway_combined_3class/oslo/6346/data')

TILES_TO_RUN = {
    #'oslo_32-1-516-134-27': OSLO_RAW / '32-1-516-134-27.laz',
    #'oslo_32-1-516-135-21': OSLO_RAW / '32-1-516-135-21.laz',
    #'oslo_32-1-514-134-41': OSLO_RAW / '32-1-514-134-41.laz',
    #'oslo_32-1-514-134-53': OSLO_RAW / '32-1-514-134-53.laz',
    'oslo_32-1-514-135-13': OSLO_RAW / '32-1-514-135-13.laz',
}

for tile_id, laz_path in TILES_TO_RUN.items():
    print(f'\n{"="*60}\n  {tile_id}\n{"="*60}')
    cmd = [
        str(PYTHON),
        str(PROJECT_ROOT / 'scripts/national_pipeline.py'),
        '--tile',          tile_id,
        '--general_ckpt',  str(GENERAL_CKPT),
        '--general_data',  GENERAL_DATA,
        '--bridge_ckpt',   str(BRIDGE_CKPT),
        '--bridge_data',   BRIDGE_DATA,
        '--raw_laz',       str(laz_path),
        '--split',         'test',
        '--out_dir',       str(OUT_DIR),
        '--laz',
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f'ERROR: {tile_id} failed (exit code {result.returncode})', file=sys.stderr)
