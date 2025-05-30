import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the total demand
    total_demand = demands.sum()
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the savings for each edge
    savings = 2 * distance_matrix - distance_matrix.sum(dim=1, keepdim=True) - distance_matrix.sum(dim=0, keepdim=True) + n
    
    # Calculate the weighted savings by multiplying with normalized demands
    weighted_savings = savings * normalized_demands
    
    # Add a penalty for edges leading to the depot (should be avoided in the heuristic)
    penalty = torch.zeros_like(weighted_savings)
    penalty[torch.arange(n), torch.arange(n)] = -1e9  # Penalty for edges to the depot
    weighted_savings += penalty
    
    return weighted_savings