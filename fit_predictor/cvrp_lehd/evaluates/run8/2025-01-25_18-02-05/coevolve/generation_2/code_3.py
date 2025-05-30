import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the heuristic values based on normalized demands
    heuristics = normalized_demands.unsqueeze(1) * distance_matrix.unsqueeze(0)
    
    # Set diagonal values to negative infinity to avoid choosing the depot for a route
    heuristics = heuristics - torch.diag(torch.diag(heuristics))
    
    return heuristics