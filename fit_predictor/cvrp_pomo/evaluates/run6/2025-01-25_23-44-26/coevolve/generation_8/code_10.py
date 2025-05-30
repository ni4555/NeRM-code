import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate inverse distance heuristic
    min_distance = distance_matrix.min(dim=1)[0]
    min_distance[0] = float('inf')  # Exclude the depot from the min distance calculation
    inv_distance = 1 / (min_distance + 1e-6)  # Add a small constant to avoid division by zero

    # Incorporate load balancing
    load_balance = demands / min_distance

    # Combine heuristics
    heuristic_values = inv_distance - load_balance

    return heuristic_values