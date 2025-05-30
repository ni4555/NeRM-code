import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the inverse distance for each edge
    inv_distance = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Calculate the demand penalty for each edge
    demand_penalty = normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Combine inverse distance and demand penalty to get the heuristic values
    heuristic_values = inv_distance - demand_penalty
    
    return heuristic_values