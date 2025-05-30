import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand along each edge
    cumulative_demand = (demands.unsqueeze(1) + demands.unsqueeze(0)) / 2
    
    # Calculate the heuristic values based on the distance and cumulative demand
    # Promising edges will have lower values (negative or zero), undesirable edges will have higher values
    heuristics = distance_matrix + cumulative_demand
    
    # Normalize the heuristics to ensure they are within the specified range
    max_value = heuristics.max()
    min_value = heuristics.min()
    normalized_heuristics = (heuristics - min_value) / (max_value - min_value)
    
    return normalized_heuristics