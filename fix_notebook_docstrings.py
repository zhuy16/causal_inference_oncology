"""
One-time script: update stale docstrings / comments in all notebooks.

Changes made:
1. Replace the defunct S3 download block in the PARQUET_PATH else-branch
   with a clear error message pointing to the setup scripts.
2. Add a top-of-cell comment describing data provenance.
"""
import json
import os
import glob

BASE    = os.path.dirname(os.path.abspath(__file__))
NB_DIR  = os.path.join(BASE, 'notebooks')

# ── replacement for the else-branch that tries the 403 S3 URL ──────────────
OLD_ELSE_PATTERNS = [
    # NB02 pattern
    """\
else:
    data_dir = download_tcga_if_needed()
    raw_df = load_tcga_clinical(data_dir)
    df = build_analysis_dataset(raw_df)
    df.to_parquet(PARQUET_PATH, index=False)
    print('Dataset cached to parquet.')""",
    # NB03-06 pattern A (try/except)
    """\
else:
    try:
        df = download_and_load()
        df.to_parquet(PARQUET_PATH, index=False)
        print(f'Dataset ready: {len(df):,} patients')
    except Exception as e:
        print(f'Download failed ({e}). Using synthetic TCGA-like data.')
        df = make_synthetic_tcga(n=8000)
        df.to_parquet(PARQUET_PATH, index=False)
        print(f'Synthetic dataset ready: {len(df):,} patients')""",
    # NB03-06 pattern B (plain else)
    """\
else:
    df = download_and_load()
    df.to_parquet(PARQUET_PATH, index=False)
    print(f'Dataset cached.')""",
]

NEW_ELSE = """\
else:
    raise FileNotFoundError(
        "analysis_dataset.parquet not found.\\n"
        "Run the setup scripts from the repo root first:\\n"
        "  python fetch_lfs_clinical.py   # download real TCGA files\\n"
        "  python build_real_dataset.py   # build the parquet cache\\n"
        "Or for offline use:\\n"
        "  python generate_synthetic_data.py"
    )"""

# ── provenance comment to prepend to every data-loading cell ───────────────
PROVENANCE_COMMENT = (
    "# Data is loaded from data/processed/analysis_dataset.parquet\n"
    "# Built by build_real_dataset.py from real TCGA Pan-Cancer Atlas 2018 clinical files.\n"
    "# Run fetch_lfs_clinical.py + build_real_dataset.py to create it,\n"
    "# or generate_synthetic_data.py for offline use.\n"
)

fixed_nbs = 0
for nb_path in sorted(glob.glob(os.path.join(NB_DIR, '0*.ipynb'))):
    with open(nb_path) as f:
        nb = json.load(f)

    changed = False
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])

        # 1. Replace stale else-branch
        for old_else in OLD_ELSE_PATTERNS:
            if old_else in src:
                src = src.replace(old_else, NEW_ELSE)
                changed = True

        # 2. Add provenance comment if PARQUET_PATH is in cell and comment not yet there
        if 'PARQUET_PATH' in src and 'Built by build_real_dataset' not in src:
            src = PROVENANCE_COMMENT + src
            changed = True

        if changed:
            lines = src.splitlines(keepends=True)
            # ensure last line has no trailing newline in JSON source list
            cell['source'] = [ln if ln.endswith('\n') else ln + '\n' for ln in lines]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')

    if changed:
        with open(nb_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f'Updated: {os.path.basename(nb_path)}')
        fixed_nbs += 1

print(f'\nDone. {fixed_nbs} notebooks updated.')
