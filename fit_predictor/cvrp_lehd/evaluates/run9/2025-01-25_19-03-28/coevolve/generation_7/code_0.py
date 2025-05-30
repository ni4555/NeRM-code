import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the demand-to-capacity ratio for each customer
    demand_to_capacity_ratio = demands / demands.sum()
    
    # Calculate the heuristic value for each edge
    heuristics = -distance_matrix * demand_to_capacity_ratio
    
    # Normalize the heuristics to be between -1 and 1
    heuristics = (heuristics - heuristics.min()) / (heuristics.max() - heuristics.min())
    
    return heuristics