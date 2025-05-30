import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize the values
    total_demand = torch.sum(demands)
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the negative of the normalized demands to create heuristics
    # Negative values for undesirable edges, positive for promising ones
    heuristics = -normalized_demands
    
    return heuristics