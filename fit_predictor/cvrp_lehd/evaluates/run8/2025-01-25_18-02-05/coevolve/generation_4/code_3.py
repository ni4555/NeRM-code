import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize customer demands
    normalized_demands = demands / total_capacity
    
    # Compute the heuristics based on normalized demands
    heuristics = normalized_demands * distance_matrix
    
    # Enforce negative values for undesirable edges and positive values for promising ones
    # This can be done by setting the heuristics to be negative if the distance is greater than 1
    # and positive otherwise. This assumes that the distance matrix contains only positive values.
    heuristics[distance_matrix > 1] *= -1
    
    return heuristics