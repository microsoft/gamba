import numpy as np
import os
from sklearn.metrics import mean_squared_error
import torch

def analyze_training_data(train_dir: str):
    """Analyze phyloP scores and calculate MSE for training data."""
    total_conservation = 0
    total_length = 0
    total_mse = 0
    n_files = 0
    
    for filename in os.listdir(train_dir):
        if filename.endswith('.npz'):
            data = np.load(os.path.join(train_dir, filename))
            conservation = data['conservation']
            sequence = data['sequence']
            
            # Calculate average phyloP
            total_conservation += np.sum(conservation)
            total_length += len(conservation)
            
            # Calculate MSE between adjacent positions
            # Convert to PyTorch tensors since you use torch
            cons_tensor = torch.tensor(conservation[:-1])
            next_cons_tensor = torch.tensor(conservation[1:])
            mse = mean_squared_error(cons_tensor, next_cons_tensor)
            total_mse += mse
            n_files += 1
            
    avg_phylop = total_conservation / total_length
    avg_mse = total_mse / n_files
    
    return avg_phylop, avg_mse

# Example usage
train_dir = "/home/mica/gamba/data_processing/data/240-mammalian/train"
avg_phylop, avg_mse = analyze_training_data(train_dir)
print(f"Average phyloP score: {avg_phylop:.4f}")
print(f"Average MSE between adjacent positions: {avg_mse:.4f}")

#Average phyloP score: 0.2511
#Average MSE between adjacent positions: 2.7909