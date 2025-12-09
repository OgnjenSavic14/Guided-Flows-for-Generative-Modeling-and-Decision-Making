import torch

from src.model import ConditionalUNet
from src.train import TrainerImages
from src.data import get_dataloader
from src.utils import get_device, ensure_dir, make_fid_loader
from src.sampler import sample_images

train_root_dir = "/home/pml02/datasets/ImageNet_train_64x64"
test_root_dir = "/home/pml02/datasets/ImageNet_val_64x64"

batch_size = 64
num_epochs = 10
lr = 1e-4
ensure_dir("models")

device = get_device()

print("Loading data...", flush = True)
train_loader, test_loader = get_dataloader(
    train_root=train_root_dir,
    val_root=test_root_dir,
    batch_size=batch_size,
    num_workers = 4
)

fid_loader = make_fid_loader(test_loader.dataset, n_fid=64, batch_size=batch_size)

print("Model creation...", flush = True)
# model = ConditionalUNet(
#     num_classes=1001,
#     in_channels=3,
#     model_channels=128,
#     out_channels=3,
#     num_res_blocks=3,
#     channel_mult=(1, 2, 3, 4),
#     attention_resolutions=(2, 4, 8),
#     dropout=0.0
# )

model = ConditionalUNet(
    num_classes=1001,
    in_channels=3,
    model_channels=192,
    out_channels=3,
    num_res_blocks=3,
    channel_mult=(1, 2, 3, 4),
    attention_resolutions=(2, 4, 8),
    dropout=0.1
)


print("Trainer creation...", flush = True)
trainer = TrainerImages(
    model=model,
    train_loader=train_loader,
    fid_loader=fid_loader,
    sampler_fn=sample_images,
    device=device,
    lr=lr,
    model_save_path="models/model_images_2.pt",
    fid_every=1,
    fid_samples=64
)

print("Training...", flush = True)
trainer.train(num_epochs = num_epochs)
