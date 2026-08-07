# RGBA Angle CNN

This project trains a small PyTorch CNN to predict the direction encoded by the `_r<angle>.png` suffix. Angles are in degrees in `[0, 360)` and the public prediction is rounded to one degree.

## Workflow

```bash
python crop_ring.py
python split_dataset.py
python train.py --output-dir runs/experiment_001
python predict.py data/val/<one-val-file>.png --checkpoint runs/experiment_001/best.pt
```

`crop_ring.py` rebuilds `data/processed` from `data/raw`, applies the ring mask, and writes transparent pixels as `(0, 0, 0, 0)`. It only removes old PNG files from `data/processed`; raw images are never changed.

`split_dataset.py` is the only script that operates on the dataset. It copies a seeded, angle-stratified split from `data/processed`: approximately 15% validation images, distributed across 30-degree angle bins with at least one validation image per bin, to `data/val`, and the rest to `data/train`. The split is recorded in `data/split_manifest.json` and reused on later runs. Use `python split_dataset.py --resplit` only when intentionally creating a new split.

`train.py` only trains: it reads `data/train` and `data/val`, never copies, moves, or splits images. Training defaults to CPU or CUDA automatically, output directory `runs/spatial-rotation`, batch size 32, zero data-loader workers, 16 CPU threads, and 400 maximum epochs with early stopping (patience 40) and ReduceLROnPlateau (patience 20). Use `--device cpu`, `--output-dir runs/experiment_name`, `--epochs N`, `--batch-size N`, or `--workers N` to override settings. The `--smoke` flag runs exactly one epoch through the normal path for verification; it is not a substitute for full training.

Resume an interrupted run from the default output directory or a specific checkpoint:

```bash
python train.py --resume
python train.py --resume runs/experiment_name/last.pt --output-dir runs/experiment_name
```

Each output directory contains `best.pt`, `last.pt`, `history.json`, and `config.json`. Start a fresh experiment in an empty output directory; use `--resume` instead of silently overwriting an existing checkpoint. The model has 995,952 trainable parameters, preserves a 4x4 coarse spatial layout before its regression head, accepts `4x112x112` RGBA input scaled to `[0, 1]`, and regresses sine/cosine components that are normalized when decoded. Validation metrics use circular errors, so the 0/360 boundary is continuous.

Training augments the training images with RGB-noise (σ=0.02) plus, with 50% probability, one clockwise rotation selected uniformly from the 24 non-zero 15-degree multiples (15°–345°). The target angle is increased by the same rotation amount modulo 360; validation images are never augmented. 90°-multiples rotate losslessly via `np.rot90`; other angles use PIL BICUBIC (clockwise = `-angle`, since PIL rotates counter-clockwise).

`predict.py` accepts exactly one existing 112x112 RGBA PNG and does not resize or convert it. It defaults to `runs/spatial-rotation/best.pt`; pass `--checkpoint` and `--device` when needed.

## Data directories

- `data/raw`: source screenshots; never modified by the scripts.
- `data/processed`: regenerated, ring-cropped RGBA images.
- `data/train`: copied training images (no augmentation applied on disk).
- `data/val`: copied held-out validation images.
- `data/split_manifest.json`: reproducible split record.
- `runs/`: checkpoints and JSON experiment results.
