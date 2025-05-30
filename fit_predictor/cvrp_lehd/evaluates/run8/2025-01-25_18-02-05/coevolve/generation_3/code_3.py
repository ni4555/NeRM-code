import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum
    
    # Calculate heuristics as a function of the inverse of the distance and the normalized demand
    heuristics = (1 / distance_matrix) * normalized_demands
    
    # Apply a simple penalty to large distances to make them less promising
    # This can be adjusted or removed based on the problem specifics
    heuristics[distance_matrix == float('inf')] = -float('inf')
    
    return heuristics