import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Compute the heuristic values as the product of the distance and the normalized demand
    # This encourages visiting nodes with higher demand (normalized demand) at shorter distances
    heuristics = distance_matrix * normalized_demands
    
    # We add a small constant to avoid division by zero when taking the reciprocal
    epsilon = 1e-8
    heuristics = heuristics / (heuristics + epsilon)
    
    # Apply a penalty to undesirable edges by setting their heuristics to a negative value
    # For example, we can use a very large negative number to represent an edge that should not be taken
    penalty = -1e9
    heuristics[distance_matrix == 0] = penalty  # Avoid division by zero with depot edges
    
    return heuristics