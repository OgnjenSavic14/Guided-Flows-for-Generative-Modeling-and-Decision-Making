import matplotlib.pyplot as plt
import numpy as np
import random
import math
from pathlib import Path
from PIL import Image
import itertools

#%matplotlib inline

def load_label_mappings(mapping_file):
    """
    Load label mappings from file.
    
    Args:
        mapping_file: path to labels file (synset_id id class_name)
    
    Returns:
        synset_to_id: dict mapping synset_id ('n02119789') to iid (1-1000)
        id_to_name: dict mapping id to class name
    """
    synset_to_id = {}
    id_to_name = {}
    
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                synset_id = parts[0]
                id = int(parts[1])
                class_name = parts[2]
                
                synset_to_id[synset_id] = id
                id_to_name[id] = class_name
    
    return synset_to_id, id_to_name


def load_data(dataset, transformation=None, n_train=None, n_test=None, data_root=None, mapping_dir=None, seed=None):
    """
    Loads the data from the dataset, applies given transformation and splits the data to the given split.
    Returns images as flattened vectors and labels as numerical values (1-1000).

    Args: 
        dataset: path to the dataset
        transformation: transformations to be applied to the data
        n_train: number of train samples
        n_test: number of test samples
        mapping_dir: path to label mapping file (required to get numerical labels)
        seed: for reproducible shuffling

    Returns:
        sample_train: lazy (vector, numerical_label) generator for training
        sample_test: lazy (vector, numerical_label) generator for testing
    """
    dataset_root = Path(data_root) / dataset / dataset if data_root else Path(dataset) / dataset
    class_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])

    if mapping_dir is not None:
        synset_to_id, _ = load_label_mappings(mapping_dir)
        class_dirs_indexed = {cls.name: synset_to_id.get(cls.name, 0) for cls in class_dirs}
    else: # just enumerate if there is no mapping (starting from 1)
        class_dirs_indexed = {cls.name: idx + 1 for idx, cls in enumerate(class_dirs)} 
    
    samples = []
    for cls_dir in class_dirs:
        label = class_dirs_indexed[cls_dir.name]
        for img_path in list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) +list(cls_dir.glob("*.png")):
            samples.append((img_path, label))

    if seed is not None:
        random.seed(seed)
    random.shuffle(samples)

    total = len(samples)
    if n_train is None and n_test is None:
        n_train = int(0.8 * total)
        n_test = total - n_train
    elif n_train is None:
        n_train = total - n_test
    elif n_test is None:
        n_test = total - n_train
    elif n_train + n_test > total:
        raise ValueError('Sample sizes combined exceed the total data size')

    train_samples = samples[:n_train]
    test_samples = samples[n_train:n_train + n_test]

    def generator(items):
        for img_path, label in items:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                data = transformation(img) if transformation else np.array(img)
                img_vector = data.flatten()
            yield img_vector, label

    return generator(train_samples), generator(test_samples)


def show(x, n, outfile=None, mapping_dir=None, title=None, img_shape=(64, 64, 3)):
    """
    Shows given number of samples and saves the output to the specifed file.
    Handles both image arrays and flattened vectors with numerical labels.
    If run another time, shows next n samples from the generator.

    Args: 
        samples: data to be represented (plain or labled images)
        rows: number of rows
        cols: number of columns
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
        _, id_to_name = load_label_mappings(mapping_dir)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    fig.suptitle(title, fontsize=20)
    axes = np.atleast_1d(axes).flatten()

    for ax, sample in zip(axes, batch):
        if isinstance(sample, tuple) and len(sample) == 2:
            data, label = sample
        else:
            data, label = sample, None

        if data.ndim == 1:
            img = data.reshape(img_shape)
        else:
            img, label = sample, None

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