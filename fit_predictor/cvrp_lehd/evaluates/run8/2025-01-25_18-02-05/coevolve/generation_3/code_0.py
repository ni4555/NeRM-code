import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Compute the heuristics as the negative of the distance multiplied by the normalized demand
    heuristics = -distance_matrix * normalized_demands
    
    return heuristics