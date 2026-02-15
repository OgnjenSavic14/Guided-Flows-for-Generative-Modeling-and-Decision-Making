import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import pickle
from pathlib import Path

from .model import ConditionalUNet
from .utils import get_myid_to_resnetid, probs_for_label, get_device, normalize_map, ensure_dir
from .data import sample_noise, sample_image_noise

@torch.no_grad()
def ode_step_rk4(model, x_t, t, h, y, w):
    y0 = torch.zeros_like(y)

    k1 = (1 - w) * model(x_t, t, y0) + w * model(x_t, t, y)
    k2 = (1 - w) * model(x_t + 0.5 * h * k1, t + 0.5 * h, y0) + w * model(x_t + 0.5 * h * k1, t + 0.5 * h, y)
    k3 = (1 - w) * model(x_t + 0.5 * h * k2, t + 0.5 * h, y0) + w * model(x_t + 0.5 * h * k2, t + 0.5 * h, y)
    k4 = (1 - w) * model(x_t + h * k3, t + h, y0) + w * model(x_t + h * k3, t + h, y)

    return x_t + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@torch.no_grad()
def sampling_from_guided_flows(model, device='cuda', y=0, w=1.0, num_steps=100, batch_size=512):
    x0 = torch.tensor(sample_noise(batch_size, 2), dtype=torch.float32, device=device)
    h = 1.0 / num_steps

    t = torch.zeros((batch_size, 1), device=device)
    x_t = x0.clone()
    y_tensor = torch.full((batch_size,), y, dtype=torch.long, device=device)

    for _ in range(num_steps):
        x_t = ode_step_rk4(model, x_t, t, h, y_tensor, w)
        t = t + h

    return x_t.detach().cpu().numpy()

@torch.no_grad()
def midpoint_solver(model: nn.Module, x_t, t, h, y, w):
    """
    Midpoint ODE step: x_{t+h} = x_t + h * u(t + h/2, x_t + h/2 * u(t))
    """
    y0 = torch.zeros_like(y)
    
    u_t = (1 - w) * model(x_t, t, y0) + w * model(x_t, t, y)
    t_mid = t + 0.5 * h
    x_mid = x_t + 0.5 * h * u_t
    
    u_mid = (1 - w) * model(x_mid, t_mid, y0) + w * model(x_mid, t_mid, y)
    
    return x_t + h * u_mid

@torch.no_grad()
def sample_images(model, noise = None, w = 1.5, device = "cuda", y=None, num_steps=100, batch_size=128, C=3, H=64, W=64):
    model.eval()

    if noise is None:
        x_t = sample_image_noise(batch_size, C, H, W, device = device)
    else:
        if not isinstance(noise, torch.Tensor):
            raise TypeError("noise must be torch.Tensor")

        if noise.shape != (batch_size, C, H, W):
            raise ValueError(
                f"Noise has shape {noise.shape}, "
                f"but requires {expected_shape}"
            )

        x_t = noise.to(device)
        
    h = 1.0 / num_steps

    t = torch.zeros(batch_size, device=device)

    if y is None:
        y = torch.zeros(batch_size, dtype=torch.long, device=device)
    elif isinstance(y, int):
        y = torch.full((batch_size,), y, dtype=torch.long, device=device)
    elif isinstance(y, torch.Tensor):
        y = y.to(device)
    else:
        raise ValueError("y must be None, int, or torch.Tensor")
    
    for i in range(num_steps):
        x_t = midpoint_solver(model, x_t, t, h, y, w)
        t = t + h
        if (i + 1) % 10 == 0:
            print(f"{i + 1}-th step completed", flush = True)
            
    x_t = torch.clamp(x_t, 0.0, 1.0)
    
    return x_t.cpu()

def generate(label, noise = None):
    device = get_device()

    model = ConditionalUNet(
        num_classes=1001,
        in_channels=3,
        out_channels=3,
        model_channels=128,
        num_res_blocks=2,
        channel_mult=(1, 2, 4),
        attention_resolutions=(8, 4),
        dropout=0.0
    )

    print("Loading Model...", flush = True)
    SRC_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_DIR.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    ensure_dir(MODELS_DIR)
    model_path = MODELS_DIR / "model_final.pt"
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    if noise is not None:
        if not isinstance(noise, torch.Tensor):
            raise TypeError("noise must be torch.Tensor")
        
        if noise.dim() == 3:
            if noise.shape != (3, 32, 32):
                raise ValueError("3D noise must have shape (3, 32, 32)")
            noise = noise.unsqueeze(0)
        elif noise.dim() == 4:
            if noise.shape != (1, 3, 32, 32):
                raise ValueError("4D noise must have shape (1, 3, 32, 32)")
        else:
            raise ValueError("Noise must have shape (3,32,32) or (1,3,32,32)")

    image = samples = sample_images(model, noise = noise, w=1.6, device=device, y=label, num_steps=200, batch_size=1, C=3, H=32, W=32)
    image = image.squeeze(0)

    SCORES_DIR = PROJECT_ROOT / "scores"
    ensure_dir(SCORES_DIR)
    scores_path = SCORES_DIR / "results_2.pkl"

    with open(scores_path, "rb") as f:
        results = pickle.load(f)

    if label not in results:
        raise ValueError(f"Label {label} confidence not calculated.")

    mean = results[label]["mean"]
    std = results[label]["std"]

    return image, mean, std


def label_confidence(model):
    myid_to_resnetid = get_myid_to_resnetid(
        "/home/pml02/Guided-Flows-for-Generative-Modeling-and-Decision-Making/dataset/label_mappings.txt",
        "/home/pml02/Guided-Flows-for-Generative-Modeling-and-Decision-Making/dataset/imagenet_class_index.json")

    device = get_device()
    print(device, flush = True)

    results = {}

    for y in range(1, 1001):
        print(f"Sampling for label = {y}", flush = True)
        images = sample_images(
            model = model,
            w = 1.6,
            y = y,
            batch_size=100,
            device=device,
            num_steps=200,
            C=3,
            H=32,
            W=32
        ).to(device)

        print("Calculating probabilities...", flush = True)
        p, max_probs, max_labels = probs_for_label(images, myid_to_resnetid[y])
        p = p.detach().cpu().numpy()
        max_probs = max_probs.detach().cpu().numpy()
        max_labels = max_labels.detach().cpu().numpy()
        mean = p.mean().round(2)
        std = p.std().round(2)

        results[y] = {
            "p": p,
            "max_probs": max_probs,
            "max_labels": max_labels,
            "mean": mean,
            "std": std
        }

    print("Saving results...", flush = True)
    with open("results_2.pkl", "wb") as f:
        pickle.dump(results, f)   


@torch.no_grad()
def midpoint_solver_h(model: nn.Module, x_t, t, h, y, w):
    """
    Midpoint ODE step: x_{t+h} = x_t + h * u(t + h/2, x_t + h/2 * u(t))
    """
    y0 = torch.zeros_like(y)
    
    u_t = (1 - w) * model(x_t, t, y0) + w * model(x_t, t, y)
    t_mid = t + 0.5 * h
    x_mid = x_t + 0.5 * h * u_t
    
    u_mid = (1 - w) * model(x_mid, t_mid, y0) + w * model(x_mid, t_mid, y)
    
    return x_t + h * u_mid, u_mid

@torch.no_grad()
def sample_images_h(model, w = 1.5, device = "cuda", y=None, num_steps=100, snapshot_ts = None, batch_size=128, C=3, H=64, W=64):
    model.eval()
    
    noise = sample_image_noise(batch_size, C, H, W, device = device)
    x_t = noise
    h = 1.0 / num_steps

    t = torch.zeros(batch_size, device=device)

    if y is None:
        y = torch.zeros(batch_size, dtype=torch.long, device=device)
    elif isinstance(y, int):
        y = torch.full((batch_size,), y, dtype=torch.long, device=device)
    elif isinstance(y, torch.Tensor):
        y = y.to(device)
    else:
        raise ValueError("y must be None, int, or torch.Tensor")

    if snapshot_ts == None:
        snapshot_ts = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
    
    heatmaps = []
    snapshots = []
    snapshot_steps = [int(round(t * num_steps)) for t in snapshot_ts]
    acc_map_raw = torch.zeros(batch_size, H, W, device=device)

    for i in range(num_steps):
        x_t, u_mid = midpoint_solver_h(model, x_t, t, h, y, w)

        # 1. Compute L1 Norm (magnitude) across channels
        # Shape becomes: (Batch, H, W)
        map_raw = u_mid.abs().sum(dim=1)
        acc_map_raw += normalize_map(map_raw)

        if i + 1 in snapshot_steps:
            heatmaps.append(normalize_map(map_raw).cpu())
            snapshots.append(x_t.cpu())

        t = t + h
        if (i + 1) % 10 == 0:
            print(f"{i + 1}-th step completed", flush = True)

    # Normalize the accumulated map to [0, 1] for final visualization
    acc_map_final = normalize_map(acc_map_raw)

    x_t = torch.clamp(x_t, 0.0, 1.0)
    
    return x_t.cpu(), noise.cpu(), torch.stack(snapshots), torch.stack(heatmaps), acc_map_final.cpu()