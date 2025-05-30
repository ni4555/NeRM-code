import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Inverse distance heuristic: lower distances are more promising
    inverse_distance = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Demand normalization heuristic: higher demand nodes are more promising
    demand_heuristic = normalized_demands * 10  # Scale factor for demand
    
    # Combine heuristics: sum of inverse distance and demand heuristic
    combined_heuristic = inverse_distance + demand_heuristic
    
    return combined_heuristic