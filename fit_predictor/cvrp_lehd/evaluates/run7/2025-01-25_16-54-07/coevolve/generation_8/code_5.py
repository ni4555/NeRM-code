import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demands = demands / total_capacity  # Normalize demands by total capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Cumulative demand mask
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Edge feasibility mask
    for i in range(n):
        for j in range(n):
            if i != j:
                if cumulative_demand[j] - cumulative_demand[i] <= 1:
                    heuristics[i, j] = 1 - cumulative_demand[j] + cumulative_demand[i]
    
    return heuristics