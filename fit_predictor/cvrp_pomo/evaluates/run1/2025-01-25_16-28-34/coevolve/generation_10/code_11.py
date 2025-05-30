import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the normalized demand difference for each edge
    normalized_demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Calculate the cost for each edge, considering the absolute difference in demand
    cost = torch.abs(normalized_demand_diff)
    
    # Incorporate the distance into the heuristic
    heuristics = cost * distance_matrix
    
    return heuristics