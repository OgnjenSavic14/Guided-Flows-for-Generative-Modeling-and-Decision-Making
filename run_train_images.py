import torch
from torch.utils.data import DataLoader

from src.model import ConditionalUNet
from src.train import TrainerImages
from src.data import get_dataloader
from src.utils import get_device, ensure_dir

# Putanja do foldera sa 64x64 ImageNet slikama
root_dir = "/home/pml02/datasets/ImageNet_train_64x64"

batch_size = 32
num_steps = 300
lr = 1e-4
ensure_dir("models")

device = get_device()

print("Loading data...", flush = True)
train_loader, test_loader = get_dataloader(
    root=root_dir,
    batch_size=batch_size,
    transform=None,
    n_train=None,
    n_test=None,
)

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



print("Trainer creation...", flush = True)
trainer = TrainerImages(
    model=model,
    dataloader=train_loader,
    device=device,
    lr=lr,
    model_save_path="models/model_images_test.pt"
)

print("Training...", flush = True)
trainer.train(num_steps=num_steps)
