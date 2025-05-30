import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    vehicle_capacity = demands[0]  # Assuming the first demand is the vehicle capacity
    normalized_demands = demands / vehicle_capacity

    # Calculate the inverse distance heuristic
    inverse_distance = 1 / distance_matrix

    # Incorporate demand penalty based on normalized demand and distance
    demand_penalty = torch.where(normalized_demands > 1, 
                                 (normalized_demands - 1) * 1000, 
                                 torch.zeros_like(normalized_demands))
    cost_with_penalty = inverse_distance - demand_penalty

    # Return the heuristics matrix
    return cost_with_penalty