import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate inverse distance heuristic
    inverse_distance = 1 / (distance_matrix + 1e-10)  # Adding a small constant to avoid division by zero

    # Calculate demand-based heuristic
    demand_heuristic = normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)

    # Combine heuristics
    combined_heuristic = inverse_distance + demand_heuristic

    # Normalize the combined heuristic to ensure it has a good balance between inverse distance and demand
    max_combined_heuristic = combined_heuristic.max()
    min_combined_heuristic = combined_heuristic.min()
    normalized_combined_heuristic = (combined_heuristic - min_combined_heuristic) / (max_combined_heuristic - min_combined_heuristic)

    return normalized_combined_heuristic