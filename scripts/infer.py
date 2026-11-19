"""Reconstruct from a sampled six-channel measurement or a folder of RGB patches."""
import argparse
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "optical"))
from models.MIMOUNet_Aux import build_net


def save_rgb(prediction, path):
    data = prediction[0].detach().cpu().numpy()
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite reconstruction")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data)
    preview = np.rint(data.clip(0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(preview).save(path.with_suffix(".png"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["ours"], default="ours")
    p.add_argument("--factor", type=int, choices=[12, 16], required=True)
    p.add_argument("--weights", type=Path, required=True)
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument("--input", type=Path, help="Sampled six-channel .npy measurement")
    sources.add_argument("--rgb-dir", type=Path, help="Folder of 960 x 960 RGB tissue patches")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, help="Output side length for a sampled measurement")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20261106)
    a = p.parse_args()
    if a.rgb_dir and a.size not in (None, 960):
        p.error("The RGB pipeline uses the 960 x 960 evaluation grid.")
    device = torch.device(a.device)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(a.weights, map_location="cpu", weights_only=True)
    if "factor" in checkpoint and int(checkpoint["factor"]) != a.factor:
        p.error("The selected sampling factor differs from the checkpoint metadata.")
    model = build_net("MIMO-UNet")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    with torch.inference_mode():
        if a.input:
            data = np.load(a.input, allow_pickle=False)
            if data.ndim == 3 and data.shape[0] != 6 and data.shape[-1] == 6:
                data = data.transpose(2, 0, 1)
            if data.ndim != 3 or data.shape[0] != 6 or data.shape[1] != data.shape[2]:
                p.error("Expected a square six-channel CHW or HWC measurement.")
            if not np.isfinite(data).all():
                p.error("Input contains nonfinite values.")
            if data.dtype == np.uint8:
                data = data.astype(np.float32) / 255
            size = a.size if a.size is not None else data.shape[-1] * a.factor
            if size <= 0 or size % 4 or a.output.suffix.lower() != ".npy":
                p.error("Use a positive output size divisible by four and a .npy output path.")
            x = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))[None].to(device)
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            save_rgb(model(x)[-1], a.output)
            return

        import imgaug
        from aligned_two_layer import AlignedTwoLens
        from data.data_load import TrainDeblurDataset
        imgaug.seed(a.seed)
        lens = AlignedTwoLens().to(device)
        lens.load_state_dict(checkpoint["lens_module"], strict=True)
        patches = TrainDeblurDataset(str(a.rgb_dir), use_transform=False, is_test=True, psf_gene_obj=lens)
        if not len(patches):
            p.error("The RGB input directory is empty.")
        stems = [Path(name).stem for name in patches.image_list]
        if len(set(stems)) != len(stems):
            p.error("Use distinct input filename stems for paired evaluation.")
        for index in range(len(patches)):
            encoded, _, filename = patches[index]
            low = F.interpolate(encoded[None], scale_factor=1 / a.factor, mode="bilinear",
                                align_corners=False, recompute_scale_factor=False)
            up = F.interpolate(low, size=(960, 960), mode="bilinear", align_corners=False)
            save_rgb(model(up)[-1], a.output / (Path(filename).stem + ".npy"))


if __name__ == "__main__":
    main()
