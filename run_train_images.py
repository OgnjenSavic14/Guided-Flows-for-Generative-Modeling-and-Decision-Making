import torch

from src.model import ConditionalUNet
from src.train import TrainerImages
from src.data import get_dataloader
from src.utils import get_device, ensure_dir, make_fid_loader
from src.sampler import sample_images

train_root_dir = "/home/pml02/datasets/ImageNet_train_32x32"
test_root_dir = "/home/pml02/datasets/ImageNet_val_32x32"

batch_size = 1024
num_epochs = 500
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

# fid_loader = make_fid_loader(test_loader.dataset, n_fid=64, batch_size=batch_size)

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

# model = ConditionalUNet(
#     num_classes=1001,
#     in_channels=3,
#     model_channels=192,
#     out_channels=3,
#     num_res_blocks=3,
#     channel_mult=(1, 2, 3, 4),
#     attention_resolutions=(2, 4, 8),
#     dropout=0.1
# )

model = ConditionalUNet(
    num_classes=1001,
    in_channels=3,
    out_channels=3,
    model_channels=128,
    num_res_blocks=2,
    channel_mult=(1, 2, 4),
    attention_resolutions=(8, 4),
    dropout=0.0
)

# CONTINUE TRAINING
print("Loading Model...", flush = True)
model.load_state_dict(torch.load("models/model_test_4.pt", map_location=device))
model.to(device)

print("Trainer creation...", flush = True)
trainer = TrainerImages(
    model=model,
    train_loader=train_loader,
    fid_loader=test_loader,
    p_unconditional = 0.2,
    sampler_fn=sample_images,
    device=device,
    lr=lr,
    model_save_path="models/model_test_4.pt",
    fid_every=1,
    fid_samples=40
)

print("Training...", flush = True)
trainer.train(num_epochs = num_epochs)
