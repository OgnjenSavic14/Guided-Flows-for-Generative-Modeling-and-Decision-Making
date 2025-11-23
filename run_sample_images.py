import torch
from src.model import ConditionalUNet
from src.sampler import sample_images
from src.utils import get_device, ensure_dir
import torchvision
import os

ensure_dir("images")
device = get_device()

batch_size = 16
num_steps = 300
label = 35

print("Model creation...", flush = True)
model = ConditionalUNet(
    num_classes=1000,
    in_channels=3,
    model_channels=128,
    out_channels=3,
    num_res_blocks=3,
    channel_mult=(1, 2, 3, 4),
    attention_resolutions=(2, 4, 8),
    dropout=0.0
)

print("Loading Model...", flush = True)
model.load_state_dict(torch.load("models/model_images_final.pt", map_location=device))
model.to(device)

print("Sampling...", flush = True)
samples = sample_images(model, device=device, y=label, num_steps=num_steps, batch_size=batch_size)

print("Saving...", flush = True)
for i in range(batch_size):
    torchvision.utils.save_image(samples[i], f"images/sample_{i}.png")  # [-1,1] -> [0,1]
