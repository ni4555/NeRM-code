import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand normalized by the vehicle capacity
    total_demand = demands.sum()
    
    # Calculate the maximum and minimum demand for normalization
    max_demand = demands.max()
    min_demand = demands.min()
    
    # Calculate the demand range and normalize demands
    demand_range = max_demand - min_demand
    normalized_demands = (demands - min_demand) / demand_range
    
    # Calculate the cost for each edge as the negative normalized demand
    cost_matrix = -normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Subtract the diagonal to avoid self-loops ( depot to itself )
    cost_matrix = cost_matrix - torch.diag(cost_matrix.diag())
    
    # Subtract the total demand from the cost matrix to prioritize
    cost_matrix = cost_matrix - total_demand
    
    return cost_matrix