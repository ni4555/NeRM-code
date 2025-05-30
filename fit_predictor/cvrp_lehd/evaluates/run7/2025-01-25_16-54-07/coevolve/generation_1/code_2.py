import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity as the sum of demands
    total_capacity = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristic values for each edge
    # A simple heuristic is to use the inverse of the demand, multiplied by the distance
    heuristics = -normalized_demands.unsqueeze(1) * distance_matrix
    
    return heuristics