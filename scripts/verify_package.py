"""Verify source or accompanying weight hashes without loading pickle objects."""
import argparse,json,hashlib
from pathlib import Path
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--weights-dir',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    rows=json.loads((root/'manifests/files_sha256.json').read_text())
    for r in rows:
        path=root/r['path']
        if not path.is_file() or digest(path)!=r['sha256']:raise RuntimeError('Hash mismatch: '+r['path'])
    print(f'Verified {len(rows)} repository files')
    if a.weights_dir:
        wr=json.loads((root/'manifests/weights_sha256.json').read_text())
        for r in wr:
            path=a.weights_dir/r['name']
            if not path.is_file() or digest(path)!=r['sha256']:raise RuntimeError('Weight mismatch: '+r['name'])
        print(f'Verified {len(wr)} weight and optical-parameter assets')
if __name__=='__main__':main()
