import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Normalize customer demands
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum
    
    # Calculate the heuristics based on the distance and normalized demand
    # A simple heuristic could be to use the negative of the distance (undesirable edges)
    # and add a positive term for the normalized demand (promising edges)
    heuristics = -distance_matrix + normalized_demands
    
    return heuristics