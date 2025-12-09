import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from .utils import load_label_mappings, check_size

def sample_mixture(batch_size=1):
    means = {
    1: np.array([-2, 0]),
    2: np.array([2, 0]),
    3: np.array([0, 2 * np.sqrt(3)])
    }
    cov = np.eye(2)
    
    y = np.random.choice([1, 2, 3], size=batch_size)
    x = np.array([np.random.multivariate_normal(means[label], cov) for label in y])
    return x, y

def sample_noise(batch_size=128, dim=2):
    return np.random.normal(size=(batch_size, dim))

def sample_time(batch_size=128):
    return np.random.uniform(0, 1, size=(batch_size,))

def sample_image_time(batch_size=128, device='cuda'):
    return torch.rand(batch_size, device=device, dtype=torch.float32)

def sample_image_noise(batch_size=128, channels=3, height=64, width=64, device='cuda'):
    return torch.randn(batch_size, channels, height, width, device=device, dtype=torch.float32)

class ImageNet64Dataset(Dataset):
    """
    Expects directory structure:
      root/
        n01440764/
          img1.jpg
          img2.png
        n01443537/
          ...
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.samples = []

        txt_files = [f for f in os.listdir(self.root) if f.lower().endswith(".txt")]
        if len(txt_files) != 1:
            raise ValueError(f"Expected exactly one .txt file in {root}, found {len(txt_files)}")
        mapping_file = os.path.join(self.root, txt_files[0])
        
        self.synset_to_id, _, _ = load_label_mappings(mapping_file)
        
        for synset in sorted(os.listdir(root)):
            synset_path = os.path.join(root, synset)
            if not os.path.isdir(synset_path):
                continue
            if synset not in self.synset_to_id:
                continue
            label1 = self.synset_to_id[synset]  # 1..1000
            # label0 = label1 - 1                 # 0..999
            for fn in os.listdir(synset_path):
                if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    # self.samples.append((os.path.join(synset_path, fn), label0))
                    self.samples.append((os.path.join(synset_path, fn), label1))

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()                # [0,1]
                # transforms.Normalize([0.5]*3, [0.5]*3) # -> [-1,1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)   # Tensor CxHxW, float32
        return img, torch.tensor(label, dtype=torch.long)

def get_dataloader(train_root, val_root = None, batch_size=128, num_workers=4, transform=None, n_train = None, n_test = None):
    ds = ImageNet64Dataset(train_root, transform)
    
    if val_root is None:
        check_size(n_train, ds, "n_train")
        check_size(n_test, ds, "n_test")
        total = len(ds)
        
        if n_train is None and n_test is None:
            n_train = int(0.8 * total)
            n_test = total - n_train
        elif n_train is None:
            n_train = total - n_test
        elif n_test is None:
            n_test = total - n_train
        else:
            check_size(n_train + n_test, ds, "n_train + n_test")

        ds_train, ds_test, _ = random_split(ds, [n_train, n_test, len(ds) - n_train - n_test])
    else:
        ds_train = ds
        ds_test = ImageNet64Dataset(val_root, transform)

        check_size(n_train, ds_train, "n_train")
        check_size(n_test, ds_test, "n_test")
        
        if n_train is not None and n_test is not None: 
            ds_train, _ = random_split(ds_train, [n_train, len(ds_train) - n_train])
            ds_test, _ = random_split(ds_test, [n_test, len(ds_test) - n_test])
        elif n_train is not None:
            ds_train, _ = random_split(ds_train, [n_train, len(ds_train) - n_train])
        elif n_test is not None:
            ds_test, _ = random_split(ds_test, [n_test, len(ds_test) - n_test])

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, test_loader 