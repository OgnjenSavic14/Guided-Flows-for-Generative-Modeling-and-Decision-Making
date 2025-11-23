import torch
from src.model import ConditionalUNet
from src.sampler import sample_images
from src.utils import get_device, ensure_dir
from dataset.dataset_utils import show
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
    model_channels=256,          # povećaj kapacitet
    out_channels=3,
    num_res_blocks=3,            # više res blokova
    channel_mult=(1,2,4,4),      # manje agresivno, stabilnije
    attention_resolutions=(32,16,8)
)

print("Loading Model...", flush = True)
model.load_state_dict(torch.load("models/model_images_test.pt", map_location=device))
model.to(device)

print("Sampling...", flush = True)
samples = sample_images(model, device=device, y=label, num_steps=num_steps, batch_size=batch_size)

print("Saving...", flush = True)
#for i in range(batch_size):
#    torchvision.utils.save_image(samples[i], f"images/sample_{i}.png")  # [-1,1] -> [0,1]

show(iter(samples), n=len(samples), mapping_dir='dataset/label_mappings.txt', 
     outfile=f'plots/{label}.png', label_for_all=label)