import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    max_demand = demands.max()
    
    # Calculate cumulative demand mask
    cumulative_demand_mask = demands.cumsum() / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate edge feasibility mask
    edge_capacity_mask = (distance_matrix < max_demand)
    
    # Update heuristics based on cumulative demand and edge feasibility
    for i in range(1, n):
        heuristics[:, i] = (cumulative_demand_mask[i] - cumulative_demand_mask[i - 1]) * edge_capacity_mask[:, i]
    
    return heuristics