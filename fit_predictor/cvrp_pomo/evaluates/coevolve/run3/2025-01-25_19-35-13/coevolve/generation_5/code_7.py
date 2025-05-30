import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristic values based on the normalized demands
    heuristics = -distance_matrix * normalized_demands
    
    # Add a small positive value to avoid zero heuristic values
    heuristics += 1e-6
    
    return heuristics