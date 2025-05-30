import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands for each node
    demand_sum = demands.sum(dim=0)
    
    # Calculate the difference between the sum of demands and the total vehicle capacity
    demand_diff = demand_sum - 1.0  # Assuming total vehicle capacity is 1 for normalization
    
    # Use the demand difference to calculate the heuristics
    # Negative values for undesirable edges (demand difference is negative)
    # Positive values for promising edges (demand difference is positive)
    heuristics = torch.where(demand_diff < 0, -demand_diff, demand_diff)
    
    return heuristics