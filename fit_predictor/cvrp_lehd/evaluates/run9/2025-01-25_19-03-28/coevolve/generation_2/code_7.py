import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized by total vehicle capacity
    demands = demands / demands.sum()
    
    # Calculate the negative of the demand as a heuristics score
    # Negative values for undesirable edges, positive for promising ones
    heuristics = -demands
    
    # Add a small constant to avoid zero values in the heuristics
    heuristics += 1e-6
    
    # Return the heuristics matrix
    return heuristics