import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of the demands for each node
    demand_sum = demands.sum(dim=0)
    
    # Calculate the maximum possible demand that can be carried by a vehicle
    max_demand = demands.max()
    
    # Calculate the difference between the total demand and the maximum demand
    demand_diff = demand_sum - max_demand
    
    # Normalize the difference by the maximum demand to get the heuristics
    heuristics = demand_diff / max_demand
    
    # Invert the sign of the heuristics to have negative values for undesirable edges
    heuristics = -heuristics
    
    return heuristics