import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristics matrix
    heuristics = distance_matrix.clone()  # Start with the distance matrix as heuristics
    
    # Subtract the demand of each customer from the distance to it
    heuristics -= normalized_demands[1:]
    
    # Normalize the heuristics to have a mean of zero and unit variance
    mean = heuristics.mean()
    std = heuristics.std()
    heuristics = (heuristics - mean) / std
    
    # Apply a penalty to edges leading to the depot (which have zero distance)
    heuristics[0, 1:] = -1e6
    heuristics[1:, 0] = -1e6
    
    return heuristics