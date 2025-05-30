import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristic values based on the normalized demands
    # For simplicity, we'll use a simple heuristic where the heuristic value is inversely proportional to the demand
    # and also take into account the distance to encourage closer customers
    heuristics = -normalized_demands * distance_matrix
    
    return heuristics