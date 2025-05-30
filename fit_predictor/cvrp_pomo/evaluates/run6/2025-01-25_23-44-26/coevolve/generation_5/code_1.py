import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to be a fraction of the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Define the demand penalty function
    # Higher penalty for edges with customers closer to vehicle capacity limit
    penalty_factor = torch.clamp(1 - normalized_demands, min=0, max=1)
    demand_penalty = penalty_factor * distance_matrix

    # Define the inverse distance heuristic
    # Smaller distance has higher heuristic value
    inverse_distance = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero

    # Combine the inverse distance heuristic and demand penalty
    heuristic_values = inverse_distance - demand_penalty

    return heuristic_values