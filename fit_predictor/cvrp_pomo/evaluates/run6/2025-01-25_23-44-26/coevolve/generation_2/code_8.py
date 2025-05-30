import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity (assumed to be 1 for simplicity)
    total_demand = demands.sum()
    
    # Compute the relative demand of each customer
    relative_demands = demands / total_demand
    
    # Calculate the heuristic value for each edge based on the relative demand
    heuristics = distance_matrix * relative_demands
    
    return heuristics