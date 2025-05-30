import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Calculate the normalized demands
    normalized_demands = demands / total_capacity
    
    # Calculate the demand difference matrix
    demand_diff_matrix = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Calculate the heuristics based on the normalized demands and the difference matrix
    # We use a simple heuristic where we penalize the edges with higher demand differences
    heuristics = -torch.abs(demand_diff_matrix)
    
    # Adjust the heuristics based on the distance matrix
    # We assume that shorter distances are more promising
    heuristics += distance_matrix
    
    # Normalize the heuristics to ensure they are within the range of the distance matrix
    heuristics /= (heuristics.max() + 1e-8)
    
    return heuristics