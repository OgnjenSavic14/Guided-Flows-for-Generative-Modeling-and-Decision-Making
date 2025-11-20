from src.model import MLP
from src.train import Trainer, TrainerConfig
from src.utils import get_device, ensure_dir

if __name__ == "__main__":
    ensure_dir("models")

    device = get_device()
    print("Using device:", device)

    config = TrainerConfig(
        batch_size=256,
        num_steps=100000,
        lr=1e-4,
    )
    model = MLP()
    trainer = Trainer(model, device, config)

    trainer.train()

