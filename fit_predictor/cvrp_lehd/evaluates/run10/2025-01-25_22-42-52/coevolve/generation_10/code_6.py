import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the heuristics
    # A simple heuristic that combines distance and demand
    # Negative values for undesirable edges and positive values for promising ones
    heuristics = -distance_matrix + normalized_demands.unsqueeze(1) * distance_matrix
    
    return heuristics