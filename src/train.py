import torch
import torch.nn as nn
import torch.optim as optim

from .data import sample_mixture, sample_noise, sample_time
from .dynamics import alpha_t, sigma_t, dalpha_dt, dsigma_dt

class TrainerConfig:
    def __init__(
        self,
        batch_size=256,
        num_steps=100000,
        lr=1e-4,
        p_unconditional=0.1,
        model_save_path="models/model_final.pt"
    ):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lr = lr
        self.p_unconditional = p_unconditional
        self.model_save_path = model_save_path


class Trainer:
    def __init__(self, model, device, config=TrainerConfig):
        self.model = model.to(device)
        self.device = device
        self.cfg = config

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self):
        bs = self.cfg.batch_size

        # 1. Sample mixture
        x1, y = sample_mixture(batch_size=bs)
        x1 = torch.tensor(x1, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)

        # conditional/unconditional mix
        mask = torch.rand(bs, device=self.device) < self.cfg.p_unconditional
        y[mask] = 0

        # 2. Noise + time
        x0 = torch.tensor(sample_noise(bs), dtype=torch.float32, device=self.device)
        t = torch.tensor(sample_time(bs), dtype=torch.float32, device=self.device).unsqueeze(1)

        # 3. x_t and dx/dt
        x_t = alpha_t(t) * x1 + sigma_t(t) * x0
        dx_dt = dalpha_dt(t) * x1 + dsigma_dt(t) * x0

        # 4. Forward
        u_t_pred = self.model(x_t, t, y)

        # 5. Loss
        loss = self.loss_fn(u_t_pred, dx_dt)

        # 6. Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        self.model.train()

        for step in range(1, self.cfg.num_steps + 1):
            loss = self.train_step()

            if step % 1000 == 0:
                print(f"Step {step}/{self.cfg.num_steps}, Loss: {loss:.6f}", flush=True)

        torch.save(self.model.state_dict(), self.cfg.model_save_path)
        print(f"Model saved to {self.cfg.model_save_path}")
