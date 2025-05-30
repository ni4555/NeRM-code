import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate cumulative demand
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Calculate the difference between cumulative demand and individual demands
    demand_diff = cumulative_demand - demands
    
    # Normalize the demand difference by the total vehicle capacity
    normalized_demand_diff = demand_diff / demands.sum()
    
    # Calculate the heuristics values
    heuristics_matrix = distance_matrix * normalized_demand_diff
    
    return heuristics_matrix