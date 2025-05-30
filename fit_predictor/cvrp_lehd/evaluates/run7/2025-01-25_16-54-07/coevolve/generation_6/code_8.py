import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand for each node
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Calculate the normalized cumulative demand
    total_capacity = demands[-1]  # Assuming the last demand is the vehicle capacity
    normalized_cumulative_demand = cumulative_demand / total_capacity
    
    # Calculate the potential increase in demand at each edge
    potential_demand_increase = normalized_cumulative_demand[1:] - normalized_cumulative_demand[:-1]
    
    # Calculate the heuristics value as the negative potential demand increase
    # This encourages us to prioritize edges that reduce the cumulative demand increase
    heuristics = -potential_demand_increase
    
    return heuristics