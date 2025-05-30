import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative sum of demands
    total_demand = demands.sum()
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Compute the heuristic values
    heuristics = distance_matrix.clone()
    heuristics[0, :] = -float('inf')  # No edge from depot to itself
    heuristics[:, 0] = -float('inf')  # No edge from any customer to depot
    
    # Add positive heuristic values for edges that increase cumulative demand
    for i in range(1, len(demands)):
        heuristics[i, :] = heuristics[i, :] + (cumulative_demand[i] - cumulative_demand[i - 1])
    
    # Normalize the heuristics by the total demand
    heuristics /= total_demand
    
    return heuristics