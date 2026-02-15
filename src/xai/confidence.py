import torch
import torch.nn.functional as F
import torchvision.models as models
import pickle
import json

from ..utils import get_device, load_label_mappings
from ..imagenet.sampler import sample_images

def get_myid_to_resnetid(mapping_file, resnet_mapping):
    synset_to_id, _, _ = load_label_mappings(mapping_file)

    with open(resnet_mapping, "r") as f:
        resnet_map = json.load(f)
        
    myid_to_resnetid = {}

    for res in resnet_map.keys():
        myid_to_resnetid[synset_to_id[resnet_map[res][0]]] = int(res)

    return myid_to_resnetid

def preprocess_image(x: torch.Tensor) -> torch.Tensor:
    device = get_device()
    x = x.to(device)

    if x.dtype != torch.float32:
        x = x.float()
    if x.max() > 1.5:
        x = x / 255.0

    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    x = (x - mean) / std
    
    return x

@torch.no_grad()
def probs_for_label(images: torch.Tensor, label) -> torch.Tensor:
    device = get_device()
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    model.eval()

    B = images.size(0)

    if isinstance(label, int):
        label = torch.full((B,), label, dtype=torch.long, device=device)
    else:
        label = label.to(device).long()
        if label.dim() == 0:
            label = label.expand(B)    

    x = preprocess_image(images)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    
    p = probs.gather(1, label.view(-1,1)).squeeze(1)
    max_probs, max_labels = probs.max(dim=1)  # top1
    return p, max_probs, max_labels

def label_confidence(model):
    myid_to_resnetid = get_myid_to_resnetid(
        "/home/pml02/Guided-Flows-for-Generative-Modeling-and-Decision-Making/dataset/label_mappings.txt",
        "/home/pml02/Guided-Flows-for-Generative-Modeling-and-Decision-Making/dataset/imagenet_class_index.json")

    device = get_device()
    print(device, flush = True)

    results = {}

    for y in range(1, 1001):
        print(f"Sampling for label = {y}", flush = True)
        images = sample_images(
            model = model,
            w = 1.6,
            y = y,
            batch_size=100,
            device=device,
            num_steps=200,
            C=3,
            H=32,
            W=32
        ).to(device)

        print("Calculating probabilities...", flush = True)
        p, max_probs, max_labels = probs_for_label(images, myid_to_resnetid[y])
        p = p.detach().cpu().numpy()
        max_probs = max_probs.detach().cpu().numpy()
        max_labels = max_labels.detach().cpu().numpy()
        mean = p.mean().round(2)
        std = p.std().round(2)

        results[y] = {
            "p": p,
            "max_probs": max_probs,
            "max_labels": max_labels,
            "mean": mean,
            "std": std
        }

    print("Saving results...", flush = True)
    with open("results_2.pkl", "wb") as f:
        pickle.dump(results, f)   
