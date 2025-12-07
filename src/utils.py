import torch
import os
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np

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