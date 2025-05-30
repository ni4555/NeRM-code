import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize customer demands
    normalized_demands = demands / total_capacity
    
    # Calculate the average distance to use as a base for heuristic
    average_distance = distance_matrix.mean()
    
    # Compute heuristic values
    # Use a simple heuristic based on the ratio of demand to distance from the depot
    # More demanding nodes and those closer to the depot are considered more promising
    heuristics = normalized_demands * distance_matrix / average_distance
    
    # Adjust heuristic values to ensure they are negative for undesirable edges and positive for promising ones
    heuristics = heuristics - heuristics.max()
    
    return heuristics