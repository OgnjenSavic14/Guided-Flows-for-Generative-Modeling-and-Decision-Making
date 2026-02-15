import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from src.imagenet.model import ConditionalUNet
from src.imagenet.sampler import sample_images
from src.utils import get_device, ensure_dir, show
import torchvision
import os

ensure_dir("images")
device = get_device()
print(device, flush = True)

batch_size = 16
num_steps = 200
label = 450

print("Model creation...", flush = True)
# model = ConditionalUNet(
#     num_classes=1001,
#     in_channels=3,
#     model_channels=128,
#     out_channels=3,
#     num_res_blocks=3,
#     channel_mult=(1, 2, 3, 4),
#     attention_resolutions=(2, 4, 8),
#     dropout=0.0
# )

# model = ConditionalUNet(
#     num_classes=1001,
#     in_channels=3,
#     model_channels=192,
#     out_channels=3,
#     num_res_blocks=3,
#     channel_mult=(1, 2, 3, 4),
#     attention_resolutions=(2, 4, 8),
#     dropout=0.1
# )

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

print("Sampling...", flush = True)
samples = sample_images(model, w=1.5, device=device, y=label, num_steps=num_steps, batch_size=batch_size, C=3, H=32, W=32)

print("Saving...", flush = True)
for i in range(batch_size):
   torchvision.utils.save_image(samples[i], f"images/sample_test_4_{i}.png")

show(iter(samples), n=len(samples), mapping_dir='dataset/label_mappings.txt', 
     outfile=f'plots/test__4_{label}.png', label_for_all=label)
