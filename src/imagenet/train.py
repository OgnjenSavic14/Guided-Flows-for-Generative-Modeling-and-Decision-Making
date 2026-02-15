import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torchmetrics.image.fid import FrechetInceptionDistance

from .data import sample_image_noise, sample_image_time
from ..dynamics import alpha_t, sigma_t, dalpha_dt, dsigma_dt
from .sampler import sample_images

import psutil, os

process = psutil.Process(os.getpid())

def gpu_mem():
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_alloc = torch.cuda.max_memory_allocated() / 1e9
    return alloc, reserved, max_alloc

def cpu_mem():
    return process.memory_info().rss / 1e9

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
        self.scaler = torch.cuda.amp.GradScaler()

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=200
        )

        self.fid = FrechetInceptionDistance(feature=2048).to(self.device)
        self.fid_every = fid_every
        self.fid_samples = fid_samples

    def train_step(self, x1, y):
        self.model.train()
        B, C, H, W = x1.shape
        
        x1 = x1.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        mask = torch.rand_like(y, dtype=torch.float, device=y.device) < self.p_unconditional
        y[mask] = 0

        # 2. Noise + time
        x0 = sample_image_noise(batch_size=B, channels=C, height=H, width=W, device=self.device)
        t = sample_image_time(batch_size=B, device=self.device)

        t_b = t.view(B, 1, 1, 1)

        # 3. x_t and dx/dt
        x_t = alpha_t(t_b) * x1 + sigma_t(t_b) * x0
        dx_dt = dalpha_dt(t_b) * x1 + dsigma_dt(t_b) * x0
        
        # 4. Forward + Loss
        with torch.cuda.amp.autocast():
            u_pred = self.model(x_t, t, y)
            loss = self.loss_fn(u_pred, dx_dt)
        
        # 5. Backprop
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss

    @torch.no_grad()
    def compute_fid(self):
        self.model.eval()
        self.fid.reset()

        real_count = 0
        fake_count = 0

        # --- 1) Real images ---
        for real_imgs, _ in self.fid_loader:
            real_imgs = real_imgs.to(self.device)

            if real_count + real_imgs.size(0) > self.fid_samples:
                needed = self.fid_samples - real_count
                real_imgs = real_imgs[:needed]

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
                y = 449,
                batch_size=batch,
                device=self.device,
                num_steps=100,
                C=3,
                H=32,
                W=32
            ).to(self.device)
            
            fake_imgs_uint8 = (fake_imgs * 255).to(torch.uint8)
            self.fid.update(fake_imgs_uint8, real=False)

            fake_count += batch

        fid_value = self.fid.compute().item()
        return fid_value   

    def train(self, num_epochs=10):
        for epoch in range(1, num_epochs + 1):
            torch.cuda.reset_peak_memory_stats()
            epoch_loss = torch.tensor(0.0, device=self.device)
            num_batches = 0

            print(f"Starting epoch = {epoch}", flush = True)
            epoch_start_time = datetime.now()
            start_time = datetime.now()
            for x1, y in self.train_loader:
                loss = self.train_step(x1, y)
                
                # prev_lr = self.optimizer.param_groups[0]['lr']
                # self.scheduler.step(loss)
                # new_lr = self.optimizer.param_groups[0]['lr']
                # if new_lr < prev_lr:
                #     print(f"Learning rate decreased from {prev_lr:.2e} to {new_lr:.2e}", flush = True)
                
                epoch_loss += loss.detach()
                num_batches += 1

                if num_batches % 100 == 0:
                    print(x1.shape, flush = True)
                    alloc, reserved, max_alloc = gpu_mem()
                    cpu = cpu_mem()
                    print(
                        f"[Epoch {epoch} | Batch {num_batches}] "
                        f"GPU alloc={alloc:.2f}GB | "
                        f"GPU peak={max_alloc:.2f}GB | "
                        f"CPU rss={cpu:.2f}GB",
                        flush=True
                    )
                    end_time = datetime.now()
                    print(f"Epoch {epoch}:  Batch {num_batches} - loss: {loss:.6f}", flush = True)
                    duration = end_time - start_time
                    print(f"                100 batches time: {duration}", flush = True)
                    start_time = datetime.now()
                
            avg_loss = epoch_loss.item() / num_batches
            print(f"Epoch {epoch}/{num_epochs} - avg loss: {avg_loss:.6f}", flush = True)         

            if epoch % self.fid_every == 0:
                print("Computing FID...", flush = True)
                fid_score = self.compute_fid()
                print(f"FID after epoch {epoch}: {fid_score:.2f}", flush = True)

            epoch_end_time = datetime.now()
            duration = epoch_end_time - epoch_start_time
            torch.save(self.model.state_dict(), self.model_save_path)
            print(f"Epoch run time: {duration}", flush = True)

        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Saved model to {self.model_save_path}", flush=True)
