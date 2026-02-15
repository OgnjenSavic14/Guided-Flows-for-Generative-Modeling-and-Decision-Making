import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.toy.model import MLP
from src.toy.sampler import sampling_from_guided_flows
from src.utils import get_device, ensure_dir


if __name__ == "__main__":
    ensure_dir("samples")
    device = get_device()
    print("Using device:", device, flush = True)

    model = MLP().to(device)
    model.load_state_dict(torch.load("models/model_final.pt", map_location=device))
    model.eval()

    print("Sampling...", flush = True)

    all_clusters = []
    n_iters = 2000
    batch_size = 512

    for i in range(3):
        y = i + 1
        cluster_batches = []

        for j in range(n_iters):
            samples = sampling_from_guided_flows(
                model, device,
                y=y, w=4.0, num_steps=100,
                batch_size=batch_size
            )
            cluster_batches.append(samples)

            if (j + 1) % 100 == 0:
                print(f"Cluster {y}, iteration {j+1}/{n_iters}", flush = True)

        all_clusters.append(np.concatenate(cluster_batches, axis=0))

    print("Saving samples", flush = True)
    np.savez("samples/points_4.npz", *all_clusters)
