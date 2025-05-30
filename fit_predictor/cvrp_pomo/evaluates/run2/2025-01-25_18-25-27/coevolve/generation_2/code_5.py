import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demands for each edge
    demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Calculate the absolute difference to avoid negative values
    abs_demand_diff = torch.abs(demand_diff)
    
    # Calculate the heuristic based on the absolute difference
    # and add the distance matrix to encourage shorter paths
    heuristics = abs_demand_diff + distance_matrix
    
    return heuristics