import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand along each edge
    cumulative_demand = demands.unsqueeze(1) + demands.unsqueeze(0)
    
    # Normalize the cumulative demand by the total vehicle capacity
    normalized_demand = cumulative_demand / demands.sum()
    
    # Calculate the heuristic as the negative of the normalized demand
    # Negative values indicate undesirable edges, positive values indicate promising ones
    heuristics = -normalized_demand
    
    return heuristics