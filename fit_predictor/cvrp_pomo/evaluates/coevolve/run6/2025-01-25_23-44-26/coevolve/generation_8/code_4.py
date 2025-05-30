import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix
    max_distance = torch.max(distance_matrix)
    distance_matrix = distance_matrix / max_distance

    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    demands = demands / total_capacity

    # Calculate inverse distance heuristic
    inv_distance = 1 / distance_matrix

    # Calculate load balancing heuristic
    load_balance = demands * inv_distance

    # Combine heuristics with a weighted sum
    combined_heuristic = (inv_distance * 0.5) + (load_balance * 0.5)

    # Apply demand penalty for edges leading to overloading
    demand_penalty = (demands > 1.0).float() * (demands - 1.0)

    # Combine the combined heuristic with the demand penalty
    final_heuristic = combined_heuristic - demand_penalty

    return final_heuristic