import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity as the sum of demands
    total_capacity = demands.sum()
    
    # Normalize customer demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Skip the diagonal (self-loops)
                # Calculate the heuristic value based on the normalized demand
                heuristics[i, j] = -normalized_demands[j]
    
    return heuristics