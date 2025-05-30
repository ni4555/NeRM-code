import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()
    
    # Normalize the customer demands
    normalized_demands = demands / vehicle_capacity
    
    # Compute the heuristics based on normalized demands
    # For simplicity, use a simple heuristic: the more the demand, the more promising the edge
    heuristics = -normalized_demands * distance_matrix
    
    return heuristics