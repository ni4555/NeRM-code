import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands to sum to 1
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum
    
    # Calculate the heuristics
    # Here, a simple heuristic could be the inverse of the demand
    # Higher demand = lower heuristic value (negative value)
    heuristics = -normalized_demands
    
    return heuristics