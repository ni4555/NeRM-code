import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands to the range [0, 1]
    demands_normalized = demands / demands.sum()
    
    # Calculate the heuristics using the demand normalized by the distance
    heuristics = demands_normalized / distance_matrix
    
    # Apply a small positive value to avoid division by zero
    epsilon = 1e-10
    heuristics = torch.clamp(heuristics, min=epsilon)
    
    return heuristics