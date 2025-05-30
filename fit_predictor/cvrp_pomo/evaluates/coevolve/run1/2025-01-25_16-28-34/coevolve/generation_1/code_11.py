import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative of the demands as a heuristic penalty for high demands
    penalty = -demands
    
    # Use the distance matrix directly as the heuristic for travel cost
    travel_cost = distance_matrix.clone()
    
    # Combine the two into a single heuristic matrix
    heuristics = penalty + travel_cost
    
    return heuristics