import torch

from .data import sample_noise

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
