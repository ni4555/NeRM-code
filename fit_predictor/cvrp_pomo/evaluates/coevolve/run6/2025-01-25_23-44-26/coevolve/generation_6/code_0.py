import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the inverse distance heuristic values
    inv_distance = 1 / (distance_matrix + 1e-6)  # Adding a small constant to avoid division by zero
    
    # Calculate the demand penalty function
    demand_penalty = (1 + normalized_demands) * (1 - demands / total_capacity)
    
    # Combine the inverse distance and demand penalty to get the heuristic values
    heuristic_values = inv_distance - demand_penalty
    
    return heuristic_values