import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the cost for each edge based on normalized demand
    demand_cost = distance_matrix * normalized_demands
    
    # Return the negative of the demand cost for heuristics (promising edges)
    heuristics = -demand_cost
    return heuristics