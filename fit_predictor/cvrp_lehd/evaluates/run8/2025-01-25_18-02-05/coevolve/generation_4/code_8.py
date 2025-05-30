import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands for normalization
    total_capacity = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_capacity
    
    # Compute the heuristic values as negative of the distance multiplied by the normalized demand
    heuristics = -distance_matrix * normalized_demands
    
    return heuristics