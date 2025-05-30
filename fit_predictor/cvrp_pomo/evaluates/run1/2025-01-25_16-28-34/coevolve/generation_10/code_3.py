import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Define a factor to adjust the heuristics (e.g., 0.1 is a small penalty for high distance)
    distance_factor = 0.1
    
    # Calculate the heuristic matrix
    # We use a negative factor for distance to promote shorter routes
    # and negative demand to promote routes with lower load
    heuristics = -distance_matrix * distance_factor - normalized_demands
    
    return heuristics