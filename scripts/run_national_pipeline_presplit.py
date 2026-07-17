"""
Runner for national pipeline on Oslo tiles already in the test split.
Skips general model preprocessing (processed files already exist).
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
#'oslo_32-1-515-133-77': OSLO_RAW / '32-1-515-133-77.laz',  # 167 052 bro-pts
   #'oslo_32-1-515-133-45': OSLO_RAW / '32-1-515-133-45.laz',  #  58 598 bro-pts wack 
    #'oslo_32-1-515-133-71': OSLO_RAW / '32-1-515-133-71.laz',  #  37 488 bro-pts bra
    #'oslo_32-1-516-134-10': OSLO_RAW / '32-1-516-134-10.laz',  #   9 849 bro-pts bra
   #'oslo_32-1-513-135-17': OSLO_RAW / '32-1-513-135-17.laz',  #     907 bro-pts
    'oslo_32-1-514-135-16': OSLO_RAW / '32-1-514-135-16.laz',  #     669 bro-pts
}

for tile_id, laz_path in TILES_TO_RUN.items():
    print(f'\n{"="*60}\n  {tile_id}\n{"="*60}')
    cmd = [
        str(PYTHON),
        str(PROJECT_ROOT / 'scripts/national_pipeline_presplit.py'),
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
