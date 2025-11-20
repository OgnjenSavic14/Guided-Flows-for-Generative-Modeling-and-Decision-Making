import torch
import os

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
