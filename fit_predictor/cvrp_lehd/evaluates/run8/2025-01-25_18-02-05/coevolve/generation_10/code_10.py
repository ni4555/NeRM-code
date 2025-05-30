import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands for normalization
    total_capacity = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristic values
    # Using the formula: -distance * demand, where demand is normalized
    heuristics = -distance_matrix * normalized_demands
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    heuristics = torch.clamp(heuristics, min=-epsilon)
    
    return heuristics