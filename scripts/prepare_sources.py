"""Prepare the pinned MIMO-UNet dependency and apply the SCO-Net changes."""
import argparse
import hashlib
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

REVISION = "5c580135ed1c03344ac9c741267324ff90b5f209"
ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "data/__init__.py": "data/__init__.py",
    "data/data_augment.py": "data/data_augment.py",
    "data/data_load.py": "data/data_load.py",
    "models/layers.py": "models/layers.py",
    "models/MIMOUNet.py": "models/MIMOUNet_Aux.py",
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, help="Existing checkout of the pinned MIMO-UNet revision")
    args = parser.parse_args()
    if args.upstream:
        head = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
        if head != REVISION:
            parser.error("The specified checkout does not match the pinned revision.")
        contents = {name: (args.upstream / name).read_bytes() for name in FILES}
    else:
        url = "https://codeload.github.com/chosj95/MIMO-UNet/zip/" + REVISION
        with urllib.request.urlopen(url, timeout=90) as response, zipfile.ZipFile(io.BytesIO(response.read())) as z:
            contents = {name: z.read("MIMO-UNet-" + REVISION + "/" + name) for name in FILES}
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name, local in FILES.items():
            dest = stage / local
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(contents[name].replace(b"\r\n", b"\n"))
        subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "--whitespace=nowarn", str(ROOT / "patches/sco-net.patch")], cwd=stage, check=True)
        import json
        checks = json.loads((ROOT / "manifests/prepared_sources.json").read_text())
        for local in FILES.values():
            if hashlib.sha256((stage / local).read_bytes()).hexdigest() != checks[local]:
                raise RuntimeError("Source verification failed for " + local)
        for local in FILES.values():
            dest = ROOT / "src/optical" / local
            if dest.exists() and dest.read_bytes() != (stage / local).read_bytes():
                raise FileExistsError("A different working copy already exists: " + str(dest))
        for local in FILES.values():
            dest = ROOT / "src/optical" / local
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(stage / local, dest)
    print("SCO-Net decoder and image loader are ready.")

if __name__ == "__main__":
    main()

