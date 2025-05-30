import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    demands_sum = demands.sum()
    normalized_demands = demands / demands_sum
    cost_matrix = distance_matrix.clone()
    
    # Add a large penalty for edges with demand of zero after normalization
    penalty = 1e5 * (1 - normalized_demands)
    cost_matrix[penalty == 1] = torch.inf
    
    # Calculate the heuristic based on the normalized demands
    heuristics = cost_matrix * normalized_demands
    
    return heuristics