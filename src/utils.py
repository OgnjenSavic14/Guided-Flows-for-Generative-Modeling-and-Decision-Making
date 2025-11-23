import torch
import os

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_label_mappings(mapping_file):
    synset_to_id = {}
    name_to_id = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                synset_id = parts[0]
                id_num = int(parts[1])
                class_name = parts[2]
                synset_to_id[synset_id] = id_num
                name_to_id[class_name] = id_num
    return synset_to_id, name_to_id

