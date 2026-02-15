import torch
from torch import nn

from .data import sample_image_noise

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
def sample_images(model, w = 1.5, device = "cuda", y=None, num_steps=100, batch_size=128, C=3, H=64, W=64):
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
        x_t = midpoint_solver(model, x_t, t, h, y, w)
        t = t + h
        if (i + 1) % 10 == 0:
            print(f"{i + 1}-th step completed", flush = True)
            
    x_t = torch.clamp(x_t, 0.0, 1.0)
    
    return x_t.cpu()
