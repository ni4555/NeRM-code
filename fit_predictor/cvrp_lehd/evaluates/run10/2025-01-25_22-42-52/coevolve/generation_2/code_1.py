import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristics using a simple approach:
    # The heuristic for an edge is the negative of the distance multiplied by the normalized demand
    heuristics = -distance_matrix * normalized_demands
    
    return heuristics