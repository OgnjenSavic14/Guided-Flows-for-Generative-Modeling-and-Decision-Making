import torch
import torch.nn as nn
import torch.optim as optim

from .data import sample_mixture, sample_noise, sample_time, sample_image_noise
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
        x0 = torch.tensor(sample_noise(bs, x1.shape[1]), dtype=torch.float32, device=self.device)
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

class TrainerImages:
    def __init__(self, model, dataloader, device='cuda', lr=1e-3, model_save_path='models/model_images_final.pt'):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.model_save_path = model_save_path

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=200
        )

    def train_step(self, x1, y):
        self.model.train()

        B, C, H, W = x1.shape
        
        x1 = x1.to(self.device)
        y = y.to(self.device)

        # 2. Noise + time
        x0 = torch.tensor(sample_image_noise(B, C, H, W), device=self.device)
        t = torch.tensor(sample_time(B), dtype=torch.float32, device=self.device)

        t_b = t.view(B, 1, 1, 1)

        # 3. x_t and dx/dt
        x_t = alpha_t(t_b) * x1 + sigma_t(t_b) * x0
        dx_dt = dalpha_dt(t_b) * x1 + dsigma_dt(t_b) * x0

        # 4. Forward
        u_pred = self.model(x_t, t, y)

        # 5. Loss
        loss = self.loss_fn(u_pred, dx_dt)

        # 6. Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, num_steps=100000):
        data_iter = iter(self.dataloader)
        
        for step in range(1, num_steps+1):
            try:
                x1, y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                x1, y = next(data_iter)

            loss = self.train_step(x1, y)

            prev_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr < prev_lr:
                print(f"Learning rate decreased from {prev_lr:.2e} to {new_lr:.2e}", flush = True)

            if step % 10 == 0:
                print(f"Step {step}/{num_steps} - loss: {loss:.6f}", flush = True)

        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Saved model to {self.model_save_path}", flush = True)