import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    demands_sum = demands.sum()
    demand_factors = demands / demands_sum
    distance_factors = distance_matrix / distance_matrix.max()
    
    # Calculate the negative of the distance factors to encourage short routes
    negative_distance_factors = -distance_factors
    
    # Calculate the heuristics by combining demand factors and negative distance factors
    heuristics = demand_factors * negative_distance_factors
    
    return heuristics