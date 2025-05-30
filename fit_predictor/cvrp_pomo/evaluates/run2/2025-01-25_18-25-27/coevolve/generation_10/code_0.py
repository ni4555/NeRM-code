import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristic values based on normalized demands and distance
    # Here we use a simple heuristic: the sum of the normalized demand and the distance
    # multiplied by a small constant to weigh the demand more than the distance.
    # This is a simplistic approach and may be replaced with more complex heuristics.
    constant = 0.1
    heuristic_values = normalized_demands * constant + distance_matrix
    
    # To make the heuristic positive for promising edges and negative for undesirable ones,
    # we subtract the minimum heuristic value from all the heuristic values.
    min_heuristic = heuristic_values.min()
    heuristic_values -= min_heuristic
    
    return heuristic_values