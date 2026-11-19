# SCO-Net optical reconstruction

Source code for *Experimental Realization of an Optical Super-Resolution Processing Network for Large-Area Cell Observation*.

The two SLMs were used to measure the front-end diffraction PSF. The tissue reconstruction uses previously scanned digital patches. The full optical layout in Fig. 12 is intended for future tissue-imaging experiments.

## Models and optical maps

The six trained assets (checkpoints, optical parameters and exported maps) accompany the manuscript in the reviewer archive. Place those files in `weights/` to reproduce the checkpoint-based results. The training entry point builds a new model from supplied RGB patches.

The ×12 and ×16 checkpoints contain the six-channel MIMO-UNet decoder and both trainable phase layers from the corresponding 2,000-update controlled run. Select the checkpoint for the desired factor. The original optical results in manuscript Table 1 were recorded separately.

For each factor, the optical-parameter NPZ stores the two unconstrained optimization arrays. The height-and-phase NPZ stores the derived layer heights in metres and wrapped phases in radians on a 1872 × 1872 grid at a pitch of 3 µm. The phase channels correspond to 640, 550 and 460 nm in that order. The arrays describe the numerical phase planes.

## Setup

Clone this repository and run the commands below from its root. Git is required. The preparation script retrieves a pinned release commit of MIMO-UNet and applies the SCO-Net decoder and data-loader changes. Install the dependencies with a Python 3.10 environment and the CUDA-enabled PyTorch build appropriate for the machine. The archived training environment used PyTorch 2.10.0 and CUDA 13.0.

```sh
python scripts/prepare_sources.py
python -m pip install -r requirements.txt
python scripts/verify_package.py
```

## Reconstruction and evaluation

For end-to-end reconstruction, supply a folder of 960 × 960 RGB patches with distinct filename stems (PNG or JPEG). The optical model generates the wavelength-specific PSFs, the archived image loader forms six input channels, and the factor-specific decoder reconstructs the sampled image. The seed controls the stochastic optical and image operations in this evaluation. Each image produces a float RGB array in CHW order and a clipped PNG preview. Paired scoring uses the float arrays and matches reference images by filename stem.

```sh
python scripts/infer.py --factor 12 --weights weights/ours_control_x12_2000.pt --rgb-dir data/reference --output outputs/x12
python scripts/evaluate.py --predictions outputs/x12 --references data/reference --output outputs/x12_metrics.json
```

For ×16 use `--factor 16`, the x16 checkpoint and a separate output directory. To run the decoder on an existing six-channel measurement, supply a sampled float32 NPY in CHW order (6 × 80 × 80 for ×12, or 6 × 60 × 60 for ×16). CPU or CUDA can be selected with `--device`.

```sh
python scripts/infer.py --factor 16 --weights weights/ours_control_x16_2000.pt --input data/measurement.npy --output outputs/prediction.npy
```

## Optical map export and training

```sh
python scripts/export_optics.py --weights weights/ours_control_x12_2000.pt --output outputs/x12_maps.npz
python scripts/train_optical.py --factor 12 --train-dir data/train --output runs/ours_x12 --steps 2000
```

The training entry point uses CUDA, a folder of RGB training patches and a new output directory. Both phase maps and the decoder are optimized from the first update. Training and evaluation settings are in `protocols/optical_control.json`. Supply your own input images. To verify locally supplied checkpoints and exported maps, run `python scripts/verify_package.py --weights-dir weights`.

## Code and provenance

`src/optical/` contains the phase propagation, intensity filtering, six-channel image loader and decoder. `manifests/` records source and asset hashes. The decoder derives from Cho et al.'s MIMO-UNet. [Upstream implementation](https://github.com/chosj95/MIMO-UNet). Component reuse information is in `THIRD_PARTY_NOTICES.md`.
