import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate inverse distance heuristic values
    inv_distance = 1.0 / (distance_matrix ** 2)
    
    # Calculate demand penalty based on normalized demand
    demand_penalty = (1 - normalized_demands) * (1 + 0.1 * demands)
    
    # Combine inverse distance and demand penalty
    heuristic_values = inv_distance - demand_penalty
    
    return heuristic_values