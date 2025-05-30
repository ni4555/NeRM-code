import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demands
    normalized_demands = demands / demands.sum()
    
    # Compute the heuristics matrix
    heuristics_matrix = -distance_matrix + (normalized_demands.unsqueeze(1) * distance_matrix)
    
    # Set the diagonal to zero (no cost to visit the depot itself)
    torch.fill_diagonal_(heuristics_matrix, 0)
    
    return heuristics_matrix