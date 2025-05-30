import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    total_capacity = demands.sum()
    demands_normalized = demands / total_capacity
    
    # Calculate the initial heuristic as the inverse of the normalized demand
    heuristics = 1 / demands_normalized
    
    # Add negative values for edges leading from the depot to itself and for edges with zero demand
    heuristics[torch.arange(n), torch.arange(n)] = -float('inf')
    heuristics[torch.arange(n), torch.arange(1, n)] *= 0.5
    heuristics[torch.arange(1, n), torch.arange(n)] *= 0.5
    
    return heuristics