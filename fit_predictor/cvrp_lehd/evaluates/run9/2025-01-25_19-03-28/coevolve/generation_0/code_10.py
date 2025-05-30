import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand difference between each pair of nodes
    normalized_demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Calculate the sum of the normalized demand difference and the distance matrix
    combined = normalized_demand_diff.abs() + distance_matrix
    
    # Use a threshold to determine the heuristics values
    # For example, edges with a sum less than 10 are considered promising
    threshold = 10
    heuristics = -combined.clamp(min=0) + combined.clamp(max=threshold)
    
    return heuristics