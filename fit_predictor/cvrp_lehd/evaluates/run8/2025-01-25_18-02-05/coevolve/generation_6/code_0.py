import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristic values for each edge
    # Here we use a simple heuristic: the negative of the distance multiplied by the normalized demand
    heuristics = -distance_matrix * normalized_demands
    
    return heuristics