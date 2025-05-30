import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Calculate the sum of demands to normalize
    sum_of_demands = demands.sum()
    
    # Normalize demands to get the ratio of demand to vehicle capacity
    normalized_demands = demands / sum_of_demands
    
    # Calculate a heuristic based on the distance and demand ratio
    # For example, we can use the negative distance to penalize longer routes
    # and the negative demand ratio to penalize routes with higher demand relative to capacity
    heuristic = -distance_matrix + normalized_demands
    
    return heuristic