import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands to have a sum of 1 for each row
    normalized_demands = demands / demands.sum()
    
    # Calculate the potential benefit of each edge as the product of the normalized demand and the distance
    heuristics = (normalized_demands.unsqueeze(1) * distance_matrix.unsqueeze(0)).sum(2)
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-8
    heuristics = heuristics / (heuristics.sum() + epsilon)
    
    return heuristics