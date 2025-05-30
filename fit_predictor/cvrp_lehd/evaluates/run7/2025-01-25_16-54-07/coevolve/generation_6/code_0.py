import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate cumulative demand
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Normalize cumulative demand by total vehicle capacity
    total_capacity = demands.sum()
    normalized_cumulative_demand = cumulative_demand / total_capacity
    
    # Calculate the heuristic values
    # Promising edges will have higher normalized cumulative demand
    # Unpromising edges will have lower or negative normalized cumulative demand
    heuristics = normalized_cumulative_demand - distance_matrix
    
    return heuristics