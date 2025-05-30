import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the heuristic values
    # We penalize high demands and high distances. We can adjust the parameters to control the penalty strength.
    demand_penalty = -demands
    distance_penalty = -distance_matrix
    
    # Combine the penalties
    heuristic_values = demand_penalty + distance_penalty
    
    return heuristic_values