import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Calculate the average demand per node
    average_demand = total_demand / len(demands)
    
    # Calculate the heuristics value for each edge
    heuristics = -distance_matrix * (demands - average_demand)
    
    # Ensure that the heuristic values are non-negative
    heuristics = torch.clamp(heuristics, min=0)
    
    return heuristics