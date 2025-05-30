import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix by the sum of demands to create a heuristic based on demand density
    demand_density = distance_matrix / demands.unsqueeze(1)
    
    # Subtract the demand density from the distance matrix to make it negative for promising edges
    heuristics = demand_density - distance_matrix
    
    # Ensure all negative values are converted to 0, to represent undesirable edges
    heuristics[heuristics < 0] = 0
    
    return heuristics