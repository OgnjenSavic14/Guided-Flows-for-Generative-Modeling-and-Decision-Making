# Guided Flows for Generative Modeling and Decision Making

Implementation of class-conditional generative flows based on the paper *Guided Flows for Generative Modeling and Decision Making*. The model learns a velocity field that transports noise into data via an ODE, using **classifier-free guidance** to control the fidelity–diversity trade-off at inference time.

## Tasks

The project is structured around three tasks:

### Task 1 — Toy 2D Gaussian Mixture

Reproduce **Figure 3** from the paper: a 2D mixture of three Gaussians where increasing the guidance weight ω sharpens the conditional modes.

- **Model**: 4-layer MLP with class embedding
- **Solver**: RK4
- **Result**: `outputs/figures/Figure_3_1.png`

### Task 2 — Class-Conditional ImageNet 32×32

Train a conditional U-Net on ImageNet (32×32, 1000 classes) and evaluate sample quality using **FID scores** across different guidance weights and NFE (number of function evaluations).

- **Model**: U-Net, 128 base channels
- **Solver**: Midpoint method
- **Results**: FID curves in `outputs/figures/fid_vs_w*.png`, generated samples in `outputs/generated/`

### Task 3 — XAI Heatmaps

Visualise *where* the model focuses during generation by computing per-pixel L1 norms of the velocity field at each ODE step, producing heatmaps that highlight semantically important regions.

- **Results**: `outputs/figures/goldfish.png`, `outputs/figures/king_penguin.png`, `outputs/figures/panda.png`

Additionally, a **label confidence** analysis evaluates all 1000 classes against a pretrained ResNet-50 (`outputs/figures/hist_mu.png`, `outputs/figures/scatter_mu_sigma.png`).

## Repository Structure

```
src/                              # Library code (importable package)
├── dynamics.py                   # Shared flow interpolation (alpha_t, sigma_t)
├── utils.py                      # Shared utilities (visualisation, plotting, helpers)
├── toy/                          # Task 1
│   ├── data.py                   # 2D Gaussian mixture sampling
│   ├── model.py                  # MLP velocity network
│   ├── sampler.py                # RK4 ODE solver
│   └── train.py                  # Trainer for toy model
└── imagenet/                     # Tasks 2 & 3
    ├── data.py                   # ImageNet dataset & dataloaders
    ├── model.py                  # Conditional U-Net wrapper
    ├── unet.py                   # U-Net architecture (from improved-diffusion)
    ├── nn_utils.py               # NN building blocks
    ├── sampler.py                # Midpoint solver, image sampling, heatmap sampling, generate()
    ├── train.py                  # Trainer with FID tracking
    └── evaluate.py               # ResNet-50 label confidence evaluation

scripts/                          # Runnable entry points (run from project root)
├── toy/
│   ├── train.py                  # Train the MLP
│   ├── sample.py                 # Sample points for different ω values
│   └── plot.py                   # Render Figure 3 from saved samples
└── imagenet/
    ├── train.py                  # Train the U-Net on ImageNet 32×32
    ├── sample.py                 # Generate class-conditional images
    ├── evaluate_fid.py           # Sweep guidance weights, compute FID
    ├── explain.py                # Generate heatmap visualisations
    └── visualize_data.py         # Preview dataset samples

outputs/                          # All generated artifacts
├── models/                       # Saved checkpoints (.pt)
├── figures/                      # Plots (Figure 3, FID curves, heatmaps, confidence)
├── generated/                    # Generated ImageNet sample images
├── samples/                      # Toy task sampling data (.npz)
└── scores/                       # FID scores (.pt) and confidence results (.pkl)

dataset/                          # Label mappings (synset ↔ id ↔ class name)
```

## Running

All scripts should be run **from the project root**:

```bash
# Task 1: Toy
python scripts/toy/train.py
python scripts/toy/sample.py
python scripts/toy/plot.py

# Task 2: ImageNet training & evaluation
python scripts/imagenet/train.py
python scripts/imagenet/sample.py
python scripts/imagenet/evaluate_fid.py

# Task 3: XAI heatmaps
python scripts/imagenet/explain.py
```

> **Note**: ImageNet scripts expect the dataset at paths configured inside each script (e.g. `/home/pml02/datasets/ImageNet_train_32x32`). Update these paths to match your environment. GPU is strongly recommended for training and sampling.

## Key Dependencies

- PyTorch (≥ 2.0)
- torchvision, torchmetrics (FID computation)
- matplotlib, numpy, Pillow
- psutil (memory monitoring during training)
