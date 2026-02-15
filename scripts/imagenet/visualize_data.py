import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.imagenet.data import get_dataloader
from src.utils import load_label_mappings, show

train_root_dir = "/home/pml02/datasets/ImageNet_train_64x64"
test_root_dir = "/home/pml02/datasets/ImageNet_val_64x64"
mapping_directory = "/home/pml02/Guided-Flows-for-Generative-Modeling-and-Decision-Making/dataset/label_mappings.txt"
batch_size = 64

print("Loading data...", flush = True)
train_loader, test_loader = get_dataloader(
    train_root=train_root_dir,
    val_root=test_root_dir,
    batch_size=batch_size,
)
print(len(train_loader.dataset), len(test_loader.dataset), flush = True)

train_imgs, train_labels = next(iter(train_loader))
train_labels = train_labels.tolist()
test_imgs, test_labels = next(iter(test_loader))
test_labels = test_labels.tolist()

show(x = iter(zip(train_imgs, train_labels)), n=len(train_imgs), mapping_dir=mapping_directory, 
     outfile=f'plots/train_samples.png')
show(x = iter(zip(test_imgs, test_labels)), n=len(test_imgs), mapping_dir=mapping_directory, 
     outfile=f'plots/test_samples.png')
