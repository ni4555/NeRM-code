import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize customer demands by the total vehicle capacity
    total_capacity = torch.sum(demands)
    normalized_demands = demands / total_capacity
    
    # Calculate the inverse distance heuristic
    inverse_distance = 1 / distance_matrix
    
    # Incorporate load balancing into the heuristic
    load_balance_factor = normalized_demands / torch.clamp(distance_matrix, min=1)
    
    # Combine the inverse distance and load balance factors
    combined_heuristic = inverse_distance + load_balance_factor
    
    # Apply a demand penalty for edges with high demand relative to distance
    demand_penalty = normalized_demands * distance_matrix
    penalized_heuristic = combined_heuristic - demand_penalty
    
    return penalized_heuristic