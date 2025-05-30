import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands by the total capacity
    normalized_demands = demands / total_capacity
    
    # Compute the negative of the normalized demands to represent undesirable edges
    negative_demands = -normalized_demands
    
    # The heuristic values are the negative of the demands, which is vectorized
    heuristic_matrix = negative_demands
    
    return heuristic_matrix