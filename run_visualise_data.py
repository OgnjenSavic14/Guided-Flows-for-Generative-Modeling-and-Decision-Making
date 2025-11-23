from src.data import get_dataloader
from src.utils import load_label_mappings, show

root_dir = "/home/pml02/datasets/ImageNet_train_64x64"
mapping_directory = "/home/pml02/Guided-Flows-for-Generative-Modeling-and-Decision-Making/dataset/label_mappings.txt"
batch_size = 16

print("Loading data...", flush = True)
train_loader, test_loader = get_dataloader(
    root=root_dir,
    batch_size=batch_size,
    transform=None,
    n_train=None,
    n_test=None,
)

train_imgs, train_labels = next(iter(train_loader))
train_labels = train_labels.tolist()
train_labels = [x + 1 for x in train_labels]
test_imgs, test_labels = next(iter(test_loader))
test_labels = test_labels.tolist()
test_labels = [x + 1 for x in test_labels]


show(x = iter(zip(train_imgs, train_labels)), n=len(train_imgs), mapping_dir=mapping_directory, 
     outfile=f'plots/train_samples.png')
show(x = iter(zip(test_imgs, test_labels)), n=len(test_imgs), mapping_dir=mapping_directory, 
     outfile=f'plots/test_samples.png')