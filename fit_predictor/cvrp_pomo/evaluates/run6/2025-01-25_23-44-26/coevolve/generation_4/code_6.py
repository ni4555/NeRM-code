import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands
    total_capacity = demands[0]  # Assuming all vehicles have the same capacity as the first vehicle
    normalized_demands = demands / total_capacity
    
    # Calculate the inverse distance
    inv_distance = 1 / (distance_matrix + 1e-8)  # Adding a small value to avoid division by zero
    
    # Calculate the demand penalty
    demand_penalty = normalized_demands * (1 - 1 / (1 + torch.exp(-0.1 * (demands - demands.mean()))))
    
    # Combine inverse distance and demand penalty
    heuristics = inv_distance - demand_penalty
    
    return heuristics