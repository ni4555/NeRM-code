import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized
    demands = demands / demands.sum()
    
    # Calculate the negative distance heuristic
    negative_distance_heuristic = -distance_matrix
    
    # Calculate the demand heuristic
    demand_heuristic = (demands.unsqueeze(0) * demands.unsqueeze(1)).triu(diagonal=1)
    
    # Combine the two heuristics
    heuristics = negative_distance_heuristic + demand_heuristic
    
    return heuristics