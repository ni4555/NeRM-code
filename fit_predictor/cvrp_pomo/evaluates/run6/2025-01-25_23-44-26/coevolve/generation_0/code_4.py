import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Compute the normalized distance matrix
    normalized_distance_matrix = distance_matrix / torch.max(distance_matrix, dim=1)[0].unsqueeze(1)
    
    # Compute the relative demand matrix
    relative_demand_matrix = demands.unsqueeze(1) / demands.unsqueeze(0)
    
    # Combine normalized distance and relative demand to compute heuristics
    heuristics = normalized_distance_matrix - relative_demand_matrix
    
    return heuristics