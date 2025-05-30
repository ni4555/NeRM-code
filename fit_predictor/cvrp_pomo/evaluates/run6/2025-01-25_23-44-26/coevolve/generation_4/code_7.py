import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize distance matrix to be in terms of inverse distance
    normalized_distance = 1.0 / (distance_matrix + 1e-8)  # Add a small constant to avoid division by zero
    
    # Normalize demands by total vehicle capacity (assumed to be 1 for simplicity)
    normalized_demands = demands / demands.sum()
    
    # Calculate the penalty for high demand customers
    demand_penalty = 1 + 0.5 * (demands - demands.mean())
    
    # Combine inverse distance and demand penalty
    combined_heuristic = normalized_distance * demand_penalty
    
    return combined_heuristic