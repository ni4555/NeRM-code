import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize customer demands by total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the inverse distance heuristic
    inverse_distance = 1 / distance_matrix
    
    # Calculate the demand normalization heuristic
    demand_normalized = normalized_demands * distance_matrix
    
    # Combine the heuristics
    combined_heuristics = inverse_distance - demand_normalized
    
    # Apply a small penalty to ensure edges with no demand or high distance are not favored
    combined_heuristics += torch.clamp(combined_heuristics, min=-1e6)
    
    return combined_heuristics