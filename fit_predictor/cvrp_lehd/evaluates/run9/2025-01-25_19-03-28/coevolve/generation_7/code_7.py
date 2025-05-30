import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are between 0 and 1 (normalized)
    demands = demands / demands.sum()
    
    # Step 1: Compute the inverse of the distance matrix
    # We use a small constant to avoid division by zero
    inv_distance = 1 / (distance_matrix + 1e-8)
    
    # Step 2: Compute the demand-based penalty
    # We use the negative of the demands to penalize higher demands
    demand_penalty = -demands
    
    # Step 3: Combine the inverse distance and the demand-based penalty
    heuristic_matrix = inv_distance + demand_penalty
    
    return heuristic_matrix