import torch
import numpy as np
import matplotlib.pyplot as plt

from src.model import ConditionalUNet
from src.utils import get_device, ensure_dir, chw_to_hwc, plot_generation_with_heatmaps
from src.sampler import sample_images_h

device = get_device()
print(device, flush = True)

batch_size = 1
num_steps = 200
label = 440

print("Model creation...", flush = True)
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

snapshot_ts = [0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 1.0]

print("Sampling...", flush = True)
sample, noise, snapshots, heatmaps, acc_heatmap = sample_images_h(
    model, w=1.6,
    device=device, 
    y=label, 
    num_steps=num_steps, 
    batch_size=batch_size,
    snapshot_ts = snapshot_ts,
    C=3, 
    H=32, 
    W=32
)

ensure_dir("plots")
savepath = "plots/king_penguin.png"

print("Ploting...", flush = True)
plot_generation_with_heatmaps(sample, noise, snapshots, heatmaps, acc_heatmap, snapshot_ts, savepath)