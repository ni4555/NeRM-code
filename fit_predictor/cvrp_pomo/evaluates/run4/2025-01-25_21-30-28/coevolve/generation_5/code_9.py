import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Normalize the distance matrix by the total capacity
    normalized_distance_matrix = distance_matrix / total_capacity
    
    # Inverse distance heuristic
    inverse_distance = 1 / (normalized_distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Normalization heuristic
    demand_diff = torch.abs(demands - demands.mean())
    
    # Combine heuristics
    combined_heuristics = inverse_distance - demand_diff
    
    return combined_heuristics