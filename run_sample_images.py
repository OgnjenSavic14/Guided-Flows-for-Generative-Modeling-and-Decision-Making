import torch
from src.model import ConditionalUNet
from src.sampler import sample_images
from src.utils import get_device, ensure_dir

ensure_dir("images")
device = get_device()

batch_size = 16
num_steps = 300
label = 76

print("Model creation...", flush = True)
model = ConditionalUNet(
    num_classes=1000,
    in_channels=3,
    model_channels=128,
    out_channels=3
)

print("Loading Model...", flush = True)
model.load_state_dict(torch.load("models/model_images_final.pt", map_location=device))
model.to(device)

print("Sampling...", flush = True)
samples = sample_images(model, device=device, y=label, num_steps=num_steps, batch_size=batch_size)
print("Shape of generated images:", samples.shape)

import torchvision
import os

print("Saving...", flush = True)
os.makedirs("generated_images", exist_ok=True)
for i in range(batch_size):
    torchvision.utils.save_image((samples[i] + 1) / 2, f"images/sample_{i}.png")  # [-1,1] -> [0,1]
