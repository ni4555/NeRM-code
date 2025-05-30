import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand along each edge
    cumulative_demand = demands[1:] + demands[:-1]
    
    # Normalize the cumulative demand by the total vehicle capacity
    normalized_demand = cumulative_demand / demands[0]
    
    # Calculate the heuristic values based on normalized demand
    # Promising edges have positive values, undesirable edges have negative values
    heuristics = -normalized_demand * distance_matrix
    
    return heuristics