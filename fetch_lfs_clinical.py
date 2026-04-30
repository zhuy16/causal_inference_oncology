"""
Download real TCGA clinical files from cBioPortal datahub LFS storage.
Replaces LFS pointer stubs with the actual TSV content.

Usage:
    python fetch_lfs_clinical.py --datahub /path/to/datahub/public

The --datahub argument should point to the `public/` subdirectory inside
your local clone of https://github.com/cBioPortal/datahub

If --datahub is omitted the script tries to find the datahub/public folder
automatically as a sibling of this repo.
"""
import argparse
import glob
import json
import os
import urllib.request

LFS_ENDPOINT = 'https://nsssw8k94d.execute-api.us-east-1.amazonaws.com/objects/batch'

parser = argparse.ArgumentParser(description='Fetch TCGA clinical LFS files.')
parser.add_argument(
    '--datahub',
    default=None,
    help='Path to the datahub/public directory (clone of github.com/cBioPortal/datahub)'
)
args = parser.parse_args()

if args.datahub:
    DATAHUB_DIR = args.datahub
else:
    # Auto-detect: look for datahub/public as a sibling of this repo
    repo_root    = os.path.dirname(os.path.abspath(__file__))
    sibling_path = os.path.join(os.path.dirname(repo_root), 'datahub', 'public')
    if os.path.isdir(sibling_path):
        DATAHUB_DIR = sibling_path
        print(f'Auto-detected datahub at: {DATAHUB_DIR}')
    else:
        print('ERROR: Could not find datahub/public directory.')
        print('Please clone https://github.com/cBioPortal/datahub and run:')
        print('  python fetch_lfs_clinical.py --datahub /path/to/datahub/public')
        raise SystemExit(1)

def read_lfs_pointer(path):
    with open(path) as f:
        content = f.read()
    if 'git-lfs' not in content:
        return None
    oid  = next(ln.split('sha256:')[1].strip() for ln in content.splitlines() if ln.startswith('oid'))
    size = int(next(ln.split()[1] for ln in content.splitlines() if ln.startswith('size')))
    return oid, size

def batch_request(objects):
    payload = json.dumps({
        'operation': 'download',
        'transfers': ['basic'],
        'objects': [{'oid': o, 'size': s} for o, s in objects]
    }).encode()
    req = urllib.request.Request(
        LFS_ENDPOINT,
        data=payload,
        headers={'Content-Type': 'application/vnd.git-lfs+json',
                 'Accept': 'application/vnd.git-lfs+json'}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())

files = sorted(glob.glob(os.path.join(DATAHUB_DIR, '*pan_can_atlas_2018',
                                       'data_clinical_patient.txt')))
print(f'Found {len(files)} clinical files.\n')

pointers = {}
for fpath in files:
    result = read_lfs_pointer(fpath)
    if result:
        pointers[fpath] = result

print(f'{len(pointers)} are LFS pointers — fetching download URLs...')
objects = list(pointers.values())
batch   = batch_request(objects)

oid_to_url = {
    obj['oid']: obj['actions']['download']['href']
    for obj in batch['objects']
    if 'actions' in obj
}

print(f'Got {len(oid_to_url)} presigned URLs. Downloading...\n')

ok = 0
for fpath, (oid, size) in pointers.items():
    cancer = os.path.basename(os.path.dirname(fpath)).split('_tcga')[0].upper()
    if oid not in oid_to_url:
        print(f'  {cancer}: no URL returned, skipping')
        continue
    urllib.request.urlretrieve(oid_to_url[oid], fpath)
    actual = os.path.getsize(fpath)
    print(f'  {cancer}: {actual:,} bytes (expected {size:,})')
    ok += 1

print(f'\nDownloaded {ok}/{len(pointers)} files.')
print('Run build_real_dataset.py next to build the parquet cache.')
