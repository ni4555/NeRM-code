import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Inverse distance heuristic
    distance_inverse = 1.0 / (distance_matrix + 1e-8)  # Add a small constant to avoid division by zero
    
    # Demand normalization heuristic
    demand_normalized = normalized_demands * distance_inverse
    
    # Combine heuristics
    combined_heuristics = demand_normalized - distance_inverse
    
    return combined_heuristics