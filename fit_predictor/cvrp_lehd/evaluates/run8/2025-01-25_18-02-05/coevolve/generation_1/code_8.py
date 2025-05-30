import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the heuristics based on normalized demands
    # Using a simple heuristic where the demand of a customer is inversely proportional to its attractiveness
    # This is a basic approach and can be replaced with more sophisticated heuristics
    heuristics = -distance_matrix * normalized_demands
    
    return heuristics