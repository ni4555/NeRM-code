import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse distance heuristic
    inv_distance = 1 / (distance_matrix + 1e-6)  # Adding a small constant to avoid division by zero
    
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the demand normalization heuristic
    demand_heuristic = normalized_demands * distance_matrix
    
    # Combine the two heuristics
    combined_heuristic = inv_distance - demand_heuristic
    
    return combined_heuristic