import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity as the sum of customer demands
    total_capacity = demands.sum()
    
    # Normalize the demands vector by the total capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the potential of each edge based on distance and demand
    potential = distance_matrix * normalized_demands.unsqueeze(1)
    
    # Introduce a penalty for high demands to discourage including those edges
    penalty = (potential > 1).float() * (potential - 1)
    
    # Add the penalty to the potential to get the heuristic value
    heuristics = potential + penalty
    
    return heuristics