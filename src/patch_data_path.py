"""
ONE-TIME utility (already applied): patched notebooks to load TCGA clinical
files from the real datahub location instead of the defunct S3 tarball URL.

This script is no longer needed for normal use — kept for reference only.
To rebuild the dataset, use fetch_lfs_clinical.py + build_real_dataset.py.
"""
import json
import glob
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(BASE, 'notebooks')
REAL_DATA_GLOB = os.path.join(BASE, 'data', 'raw', 'datahub_sparse', 'public',
                               '*pan_can_atlas_2018', 'data_clinical_patient.txt')

# ---------- patch notebooks ----------
OLD_GLOB = "glob.glob(os.path.join(extract_dir, '*', 'data_clinical_patient.txt'))"
NEW_GLOB = f"glob.glob(r'{REAL_DATA_GLOB}')"

patched = 0
for nb_path in sorted(glob.glob(os.path.join(NB_DIR, '0*.ipynb'))):
    with open(nb_path) as f:
        nb = json.load(f)
    changed = False
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        if OLD_GLOB in src:
            new_src = src.replace(OLD_GLOB, NEW_GLOB)
            cell['source'] = [ln + '\n' for ln in new_src.splitlines()]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
            changed = True
    if changed:
        with open(nb_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f'Patched: {os.path.basename(nb_path)}')
        patched += 1

print(f'\n{patched} notebooks patched.')
print(f'Real clinical files found: {len(glob.glob(REAL_DATA_GLOB))}')
print('Now re-run the notebooks — they will build the parquet from real TCGA data.')
