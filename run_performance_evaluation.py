from torchmetrics.image.fid import FrechetInceptionDistance
from src.utils import get_device
from src.data import ImageNet64Dataset
import torch
from src.model import ConditionalUNet
from src.sampler import sample_images
from datetime import datetime
from torch.utils.data import DataLoader

device = get_device()
print(device, flush = True)

batch_size = 1000
num_steps = 300
fid = FrechetInceptionDistance(feature=2048).to(device)
val_root = "/home/pml02/datasets/ImageNet_val_32x32"

ds_test = ImageNet64Dataset(val_root)
fid_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2, pin_memory=True, persistent_workers=True)

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
model.load_state_dict(torch.load("models/model_test_4.pt", map_location=device))
model.to(device)
model.eval()
fid.reset()

real_count = 0
fake_count = 0
num_samples = 10000

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
    
i = 1
print("Sampling fake images...", flush = True)
start_time = datetime.now()
while fake_count < num_samples:
    y = torch.arange(1, 1001, device=device, dtype=torch.long)
    fake_imgs = sample_images(
        model,
        w = 1.5,
        y = y,
        batch_size=batch_size,
        device=device,
        num_steps=200,
        C=3,
        H=32,
        W=32
    ).to(device)
            
    fake_imgs_uint8 = (fake_imgs * 255).to(torch.uint8)
    fid.update(fake_imgs_uint8, real=False)
    print(f"Finished batch {i}", flush = True)
    i += 1
    fake_count += batch_size
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"    Batch time: {duration}", flush = True)
    start_time = datetime.now()

print("Calculating fid...", flush = True)
fid_value = fid.compute().item()

print(f"FID score: {fid_value:.2f}", flush = True)