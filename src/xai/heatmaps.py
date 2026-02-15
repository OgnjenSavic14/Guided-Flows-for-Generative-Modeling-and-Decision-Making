import torch
from torch import nn

from ..imagenet.data import sample_image_noise
from ..utils import normalize_map

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
