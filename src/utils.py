import torch
import os
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Subset, DataLoader
import json

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

def get_myid_to_resnetid(mapping_file, resnet_mapping):
    synset_to_id, _, _ = load_label_mappings(mapping_file)

    with open(resnet_mapping, "r") as f:
        resnet_map = json.load(f)
        
    myid_to_resnetid = {}

    for res in resnet_map.keys():
        myid_to_resnetid[synset_to_id[resnet_map[res][0]]] = int(res)

    return myid_to_resnetid

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

def preprocess_image(x: torch.Tensor) -> torch.Tensor:
    device = get_device()
    x = x.to(device)

    if x.dtype != torch.float32:
        x = x.float()
    if x.max() > 1.5:
        x = x / 255.0

    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    x = (x - mean) / std
    
    return x

@torch.no_grad()
def probs_for_label(images: torch.Tensor, label) -> torch.Tensor:
    device = get_device()
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    model.eval()

    B = images.size(0)

    if isinstance(label, int):
        label = torch.full((B,), label, dtype=torch.long, device=device)
    else:
        label = label.to(device).long()
        if label.dim() == 0:
            label = label.expand(B)    

    x = preprocess_image(images)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    
    p = probs.gather(1, label.view(-1,1)).squeeze(1)
    max_probs, max_labels = probs.max(dim=1)  # top1
    return p, max_probs, max_labels