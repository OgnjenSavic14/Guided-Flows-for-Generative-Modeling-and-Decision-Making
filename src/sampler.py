import torch
import numpy as np
from torch import nn

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
def midpoint_solver(model: nn.Module, x_t, t, h, y):
    """
    Midpoint ODE step: x_{t+h} = x_t + h * u(t + h/2, x_t + h/2 * u(t))
    """
    u_t = model(x_t, t, y)
    t_mid = t + 0.5 * h
    x_mid = x_t + 0.5 * h * u_t
    u_mid = model(x_mid, t_mid, y)
    return x_t + h * u_mid

@torch.no_grad()
def sample_images(model, device = "cuda", y=None, num_steps=100, batch_size=128, C=3, H=64, W=64):
    model.eval()
    
    x_t = sample_image_noise(batch_size, C, H, W, device = device)
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
        x_t = midpoint_solver(model, x_t, t, h, y)
        t = t + h
            
    x_t = torch.clamp(x_t, 0.0, 1.0)
    
    return x_t.cpu()