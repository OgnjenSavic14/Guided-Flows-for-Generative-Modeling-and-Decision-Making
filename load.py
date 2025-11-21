from dataset.dataset_utils import load_label_mappings, load_data, show
import numpy as np

# Load training and testing data with specified transformations
train_gen, test_gen = load_data(dataset="val_blurred", 
                                transformation=lambda img: np.array(img.resize((64, 64))).astype(np.float32) / 255.0, 
                                mapping_dir='dataset/label_mappings.txt', data_root='dataset', seed=42)

# Visualize some samples from the training and testing datasets
show(train_gen, n=16, mapping_dir='dataset/label_mappings.txt', title='Training Samples', 
     outfile='plots/training_samples.png')
show(test_gen, n=16, mapping_dir='dataset/label_mappings.txt', title='Testing Samples', 
     outfile='plots/testing_samples.png')