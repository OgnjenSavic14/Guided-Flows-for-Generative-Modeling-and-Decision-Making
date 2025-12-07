import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torchmetrics.image.fid import FrechetInceptionDistance

from .data import sample_mixture, sample_noise, sample_time, sample_image_noise
from .dynamics import alpha_t, sigma_t, dalpha_dt, dsigma_dt
from .sampler import sample_images

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
    def __init__(self, model, train_loader, fid_loader, sampler_fn, p_unconditional = 0.1, device='cuda', lr=1e-3,
                 model_save_path='models/model.pt',fid_every=1, fid_samples=2000):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.fid_loader = fid_loader
        self.sampler_fn = sampler_fn
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.model_save_path = model_save_path
        self.p_unconditional = p_unconditional

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=200
        )

        self.fid = FrechetInceptionDistance(feature=2048).to(self.device)
        self.fid_every = fid_every
        self.fid_samples = fid_samples

    def train_step(self, x1, y):
        self.model.train()

        B, C, H, W = x1.shape
        
        x1 = x1.to(self.device)
        y = y.to(self.device)

        mask = torch.rand_like(y, dtype=torch.float, device=y.device) < self.p_unconditional
        y = y.clone()
        y[mask] = 0

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

    @torch.no_grad()
    def compute_fid(self):
        self.model.eval()
        self.fid.reset()

        real_count = 0
        fake_count = 0

        # --- 1) Real images ---
        for real_imgs, _ in self.fid_loader:
            real_imgs = real_imgs.to(self.device)
            real_imgs_uint8 = (real_imgs * 255).to(torch.uint8)
            self.fid.update(real_imgs_uint8, real=True)

            real_count += real_imgs.size(0)
            if real_count >= self.fid_samples:
                break

        # --- 2) Fake images ---
        bs = 32
        needed = self.fid_samples

        while fake_count < needed:
            batch = min(bs, needed - fake_count)

            fake_imgs = self.sampler_fn(
                self.model,
                batch_size=batch,
                device=self.device,
                num_steps=100
            ).to(self.device)
            
            fake_imgs_uint8 = (fake_imgs * 255).to(torch.uint8)
            self.fid.update(fake_imgs_uint8, real=False)

            fake_count += batch

        fid_value = self.fid.compute().item()
        return fid_value   

    def train(self, num_epochs=10):
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            print(f"Starting epoch = {epoch}", flush = True)
            start_time = datetime.now()
            for x1, y in self.train_loader:
                loss = self.train_step(x1, y)
                
                # prev_lr = self.optimizer.param_groups[0]['lr']
                # self.scheduler.step(loss)
                # new_lr = self.optimizer.param_groups[0]['lr']
                # if new_lr < prev_lr:
                #     print(f"Learning rate decreased from {prev_lr:.2e} to {new_lr:.2e}", flush = True)
                
                epoch_loss += loss
                num_batches += 1

                if num_batches % 100 == 0:
                    end_time = datetime.now()
                    print(f"Epoch {epoch}:  Batch {num_batches} - loss: {loss:.6f}", flush = True)
                    duration = end_time - start_time
                    print(f"                100 batches time: {duration}", flush = True)
                    start_time = datetime.now()
                
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch}/{num_epochs} - avg loss: {avg_loss:.6f}", flush = True)

            if epoch % self.fid_every == 0:
                print("Computing FID...", flush = True)
                fid_score = self.compute_fid()
                print(f"FID after epoch {epoch}: {fid_score:.2f}", flush = True)

        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Saved model to {self.model_save_path}", flush=True)