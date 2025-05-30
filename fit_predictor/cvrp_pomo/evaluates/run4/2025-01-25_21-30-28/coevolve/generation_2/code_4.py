import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized distance matrix
    normalized_distance = distance_matrix / distance_matrix.max()
    
    # Calculate the potential benefits of each edge based on demand
    demand_heuristic = 1 - (demands / demands.sum())
    
    # Combine the normalized distance with the demand heuristic
    heuristics = normalized_distance * demand_heuristic
    
    return heuristics