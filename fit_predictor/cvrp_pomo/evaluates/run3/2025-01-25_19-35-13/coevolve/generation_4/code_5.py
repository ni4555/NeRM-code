import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix
    distance_matrix = distance_matrix / distance_matrix.max()
    
    # Calculate the demand difference between each customer and the depot
    demand_diff = demands - demands.mean()
    
    # Combine the normalized distances and demand differences to create heuristics
    heuristics = distance_matrix - torch.abs(demand_diff)
    
    # Add a small constant to avoid division by zero and to ensure positive values
    epsilon = 1e-6
    heuristics = (heuristics + epsilon) / (heuristics + epsilon).max()
    
    return heuristics