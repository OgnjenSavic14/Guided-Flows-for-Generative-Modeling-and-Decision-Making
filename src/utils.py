import torch
import os
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset, DataLoader

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_label_mappings(mapping_file):
    synset_to_id = {}
    name_to_id = {}
    id_to_name = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                synset_id = parts[0]
                id_num = int(parts[1])
                class_name = parts[2]
                synset_to_id[synset_id] = id_num
                name_to_id[class_name] = id_num
                id_to_name[id_num] = class_name
    return synset_to_id, name_to_id, id_to_name

def check_size(n, dataset, name="dataset"):
    if n is not None and n > len(dataset):
        raise ValueError(f"Sample size {name}={n} exceeds the dataset size {len(dataset)}")

def make_fid_loader(dataset, n_fid=2000, batch_size=128, seed=123, num_workers=4):
    g = torch.Generator().manual_seed(seed)
    
    perm = torch.randperm(len(dataset), generator=g)[:n_fid]
    subset = Subset(dataset, perm)

    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def show(x, n, outfile=None, mapping_dir=None, title=None, img_shape=(64, 64, 3), label_for_all=None):
    """
    Shows given number of samples and saves the output to the specifed file.
    Handles both image arrays and flattened vectors with numerical labels.
    If run another time, shows next n samples from the generator.

    Args: 
        samples: data to be represented (plain or labled images)
        n: number of samples to show
        outfile: file in which the figure will be saved (if None shows the figure directly)
        mapping_dir: path to label mapping file (to convert numerical labels to class names)
        title: name of the figure title
    """
    batch = list(itertools.islice(x, n))

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    title = title if title else "Dataset Samples"

    # Load label mappings if provided
    id_to_name = {}
    if mapping_dir is not None:
        _, _, id_to_name = load_label_mappings(mapping_dir)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    if label_for_all is not None and title=="Dataset Samples":
        title = id_to_name.get(label_for_all, str(label_for_all))
        
    fig.suptitle(title, fontsize=20)
    axes = np.atleast_1d(axes).flatten()

    for ax, sample in zip(axes, batch):
        if isinstance(sample, tuple) and len(sample) == 2:
            data, label = sample
        else:
            data, label = sample, None
        # If it's a PyTorch tensor -> convert and permute
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        # If flattened vector
        if data.ndim == 1:
            img = data.reshape(img_shape)
        # If PyTorch-like CHW (3,64,64) -> convert to HWC
        elif data.ndim == 3 and data.shape[0] in (1, 3):  
            img = np.transpose(data, (1, 2, 0))
        else:
            img = data
        
        ax.imshow(img)
        ax.axis("off")

        if label is not None:
            # Show class name if there is a mapping, otherwise show numerical label
            if id_to_name and label in id_to_name:
                ax.set_title(id_to_name[label])
            else:
                ax.set_title(str(label))

    for ax in axes[len(batch):]:
        ax.axis("off")

    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=300)
    else:
        plt.show()
    plt.close(fig)

def normalize_map(map_raw, eps=1e-6):
    """
    Normalizes heatmap to [0, 1] per sample.

    Input:
        map_raw: Tensor of shape (B, H, W) or (H, W)
    Output:
        map_norm: Tensor of same shape, normalized to [0, 1]
    """
    if map_raw.dim() == 2:
        mins = map_raw.min()
        maxs = map_raw.max()
        return (map_raw - mins) / (maxs - mins + eps)

    elif map_raw.dim() == 3:
        B = map_raw.shape[0]
        flat = map_raw.view(B, -1)
        mins = flat.min(dim=1)[0].view(B, 1, 1)
        maxs = flat.max(dim=1)[0].view(B, 1, 1)
        return (map_raw - mins) / (maxs - mins + eps)

    else:
        raise ValueError("map_raw must have shape (H, W) or (B, H, W)")

def chw_to_hwc(img_chw: torch.Tensor):
    return img_chw.permute(1, 2, 0).contiguous()

def plot_generation_with_heatmaps(sample, noise, snapshots, heatmaps, acc_heatmap, snapshot_ts, savepath):
    """
    sample:        (1,3,H,W) or (3,H,W)
    snapshots:     (S,1,3,H,W) or (S,3,H,W)
    heatmaps:      (S,1,H,W) or (S,H,W)
    acc_heatmap:   (1,H,W) or (H,W)
    """

    # --- squeeze batch dims if batch_size=1 ---
    if sample.dim() == 4 and sample.size(0) == 1:
        sample = sample.squeeze(0)  # (3,H,W)

    if noise.dim() == 4 and noise.size(0) == 1:
        noise = noise.squeeze(0)    # (3,H,W)

    if acc_heatmap.dim() == 3 and acc_heatmap.size(0) == 1:
        acc_heatmap = acc_heatmap.squeeze(0)  # (H,W)

    if snapshots.dim() == 5 and snapshots.size(1) == 1:
        snapshots = snapshots.squeeze(1)  # (S,3,H,W)
    elif snapshots.dim() == 4 and snapshots.size(0) == 1:
        snapshots = snapshots.squeeze(0)

    if heatmaps.dim() == 4 and heatmaps.size(1) == 1:
        heatmaps = heatmaps.squeeze(1)  # (S,H,W)

    # Ensure cpu for plotting
    sample = sample.detach().cpu()
    noise = noise.detach().cpu()
    snapshots = snapshots.detach().cpu()
    heatmaps = heatmaps.detach().cpu()
    acc_heatmap = acc_heatmap.detach().cpu()

    S = snapshots.shape[0]
    # --- prepare final image ---
    final_img = chw_to_hwc(sample).clamp(0, 1).numpy()
    acc_hm = acc_heatmap.clamp(0, 1).numpy()
    noise_img = chw_to_hwc(noise).clamp(0, 1).numpy()

    # --- figure layout ---
    cols = 1 + S
    # cols = max(5, S)
    fig = plt.figure(figsize=(3.0 * cols, 9))
    gs = fig.add_gridspec(nrows=3, ncols=cols, height_ratios=[1.2, 1.0, 1.0])

    # # Top-left: final image
    # ax0 = fig.add_subplot(gs[0, 0:cols//2])
    # ax0.imshow(final_img)
    # ax0.set_title("Final image")
    # ax0.axis("off")

    # Row 0: final image spans all columns
    ax_final = fig.add_subplot(gs[0, :])
    ax_final.imshow(final_img)
    ax_final.set_title("Final image")
    ax_final.axis("off")

    # # Top-right: final image + accumulated heatmap
    # ax1 = fig.add_subplot(gs[0, cols//2:cols])
    # ax1.imshow(acc_hm, cmap = "jet", vmin = 0.0, vmax = 1.0)
    # ax1.set_title("Accumulated heatmap")
    # ax1.axis("off")

    # Row 1, Col 0: starting noise
    ax_noise = fig.add_subplot(gs[1, 0])
    ax_noise.imshow(noise_img)
    ax_noise.set_title("Starting noise x_0")
    ax_noise.axis("off")

    # Row 2, Col 0: accumulated heatmap
    ax_acc = fig.add_subplot(gs[2, 0])
    ax_acc.imshow(acc_hm, cmap="jet", vmin=0.0, vmax=1.0)
    ax_acc.set_title("Accumulated heatmap")
    ax_acc.axis("off")

    # Gap column (Row 1 & 2)
    for r in [1, 2]:
        ax_gap = fig.add_subplot(gs[r, 1])
        ax_gap.axis("off")

    # # Row 2: snapshots
    # for k in range(S):
    #     ax = fig.add_subplot(gs[1, k])
    #     img = chw_to_hwc(snapshots[k]).clamp(0, 1).numpy()
    #     ax.imshow(img)
    #     ax.set_title(f"t = {snapshot_ts[k]}")
    #     ax.axis("off")

    # # Row 3: snapshot heatmaps (overlay on snapshot)
    # for k in range(S):
    #     ax = fig.add_subplot(gs[2, k])
    #     hm = heatmaps[k].clamp(0, 1).numpy()
    #     ax.imshow(hm, cmap = "jet", vmin = 0.0, vmax = 1.0)
    #     # ax.set_title(f"Heatmap {k+1}")
    #     ax.axis("off")

    # # If cols > S, hide unused axes in rows 2 and 3
    # for k in range(S, cols):
    #     fig.add_subplot(gs[1, k]).axis("off")
    #     fig.add_subplot(gs[2, k]).axis("off")

    # Row 1: snapshots
    for k in range(S):
        ax = fig.add_subplot(gs[1, 1 + k])
        img = chw_to_hwc(snapshots[k]).clamp(0, 1).numpy()
        ax.imshow(img)
        ax.set_title(f"t = {snapshot_ts[k]}")
        ax.axis("off")

    # Row 2: raw heatmaps
    for k in range(S):
        ax = fig.add_subplot(gs[2, 1 + k])
        hm = heatmaps[k].clamp(0, 1).numpy()
        ax.imshow(hm, cmap="jet", vmin=0.0, vmax=1.0)
        ax.axis("off")

    # Make a small gap using spacing instead of a full column
    fig.subplots_adjust(wspace=0.1, hspace=0.25)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()