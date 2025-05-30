import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity (assuming it's the sum of all demands)
    total_capacity = demands.sum()
    
    # Normalize demands to represent the fraction of capacity each customer requires
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristics based on normalized demands
    # We use a simple heuristic: the more demand a customer has, the more promising the edge is
    # This is a basic approach and can be replaced with more sophisticated heuristics
    heuristics = normalized_demands.unsqueeze(1) * distance_matrix.unsqueeze(0)
    
    # Add a small constant to avoid division by zero when taking the reciprocal
    heuristics = heuristics + 1e-8
    
    # Take the reciprocal of the heuristics to get a negative value for undesirable edges
    heuristics = 1 / heuristics
    
    return heuristics