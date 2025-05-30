import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands for normalization
    total_demand = demands.sum()
    
    # Normalize demands to get the demand per unit of vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics as the negative of the demands multiplied by the distance squared
    # Negative values are undesirable edges, positive values are promising ones
    heuristics = -normalized_demands.unsqueeze(1) * distance_matrix ** 2
    
    return heuristics