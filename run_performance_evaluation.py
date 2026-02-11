from torchmetrics.image.fid import FrechetInceptionDistance
from src.utils import get_device, ensure_dir
from src.data import ImageNet64Dataset
import torch
from src.model import ConditionalUNet
from src.sampler import sample_images
from datetime import datetime
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ensure_dir("scores")
device = get_device()
print(device, flush = True)

batch_size = 1000
num_steps = 200
fid = FrechetInceptionDistance(feature=2048).to(device)
val_root = "/home/pml02/datasets/ImageNet_val_32x32"

ds_test = ImageNet64Dataset(val_root)
fid_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)

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
model.load_state_dict(torch.load("models/model_final.pt", map_location=device))
model.to(device)
model.eval()

num_samples = 10000

w_values = [round(1.0 + 0.1*i, 1) for i in range(9)]  # 1.0, 1.1, ..., 1.8
fid_scores = []

for w in w_values:
    print(f"Sampling fake images with w={w}...", flush=True)
    fid.reset()

    real_count = 0
    print("Loading real images...", flush = True)
    for real_imgs, _ in fid_loader:
        real_imgs = real_imgs.to(device)

        if real_count + real_imgs.size(0) > num_samples:
            needed = num_samples - real_count
            real_imgs = real_imgs[:needed]

        real_imgs_uint8 = (real_imgs * 255).to(torch.uint8)
        fid.update(real_imgs_uint8, real=True)

        real_count += real_imgs.size(0)
        if real_count >= num_samples:
            break

    fake_count = 0
    i = 1
    print("Sampling fake images...", flush = True)
    start_time = datetime.now()
    while fake_count < num_samples:
        y = torch.arange(1, 1001, device=device, dtype=torch.long)
        fake_imgs = sample_images(
            model,
            w = w,
            y = y,
            batch_size=batch_size,
            device=device,
            num_steps=num_steps,
            C=3,
            H=32,
            W=32
        ).to(device)

        fake_imgs_uint8 = (fake_imgs * 255).to(torch.uint8)
        fid.update(fake_imgs_uint8, real=False)
        print(f"    Finished batch {i}", flush=True)
        i += 1
        fake_count += batch_size

        end_time = datetime.now()
        duration = end_time - start_time
        print(f"    Batch time: {duration}", flush=True)
        start_time = datetime.now()

    fid_value = fid.compute().item()
    print(f"FID score for w={w}: {fid_value:.2f}", flush=True)
    fid_scores.append(fid_value)

torch.save({
    "w_values": w_values,
    "fid_scores": fid_scores
}, "scores/fid_scores_NFE_200_2.pt")
print("Saved FID scores to scores/fid_scores_NFE_200_2.pt", flush=True)

plt.figure(figsize=(8,5))
plt.plot(w_values, fid_scores, marker='o')
plt.xlabel("w")
plt.ylabel("FID")
plt.title("FID score vs w")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/fid_vs_w_test.png")
print("Saved plot to plots/fid_vs_w_test.png", flush=True)
