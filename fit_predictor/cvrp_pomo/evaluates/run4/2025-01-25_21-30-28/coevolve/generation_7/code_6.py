import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    
    # Demand normalization heuristic
    normalized_demands = demands / total_capacity
    
    # Inverse distance heuristic
    inverse_distance = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Combine heuristics: normalized demand and inverse distance
    heuristics = normalized_demands.unsqueeze(1) * inverse_distance.unsqueeze(0)
    
    return heuristics