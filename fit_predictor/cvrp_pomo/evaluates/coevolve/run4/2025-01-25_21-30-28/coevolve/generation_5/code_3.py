import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Inverse distance heuristic
    inv_distance = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Normalization heuristic
    demand_penalty = normalized_demands * distance_matrix
    
    # Combine heuristics
    heuristic_values = inv_distance - demand_penalty
    
    return heuristic_values