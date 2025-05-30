import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Add a small positive value to demands to avoid division by zero
    epsilon = torch.finfo(torch.float32).eps
    adjusted_demands = demands + epsilon
    
    # Calculate the heuristic values, which are inversely proportional to the demand
    # Here we use a simple inverse demand heuristic (1 / demand)
    # We also subtract the distance to ensure the heuristic is positive for short edges
    heuristics = 1 / adjusted_demands - distance_matrix
    
    # Replace negative values with zero, as we want only positive scores
    heuristics = torch.clamp(heuristics, min=0)
    
    return heuristics