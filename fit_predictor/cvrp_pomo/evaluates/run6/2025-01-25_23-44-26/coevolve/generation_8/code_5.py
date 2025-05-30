import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize the distance matrix by the maximum distance to avoid dominance
    max_distance = torch.max(distance_matrix)
    normalized_distance_matrix = distance_matrix / max_distance

    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Inverse distance heuristic: higher inverse distance, higher heuristic value
    inverse_distance_heuristic = 1 / (normalized_distance_matrix + 1e-10)  # Adding a small constant to avoid division by zero

    # Load balancing: subtract the demand to encourage balancing
    load_balance_heuristic = normalized_demands - demands

    # Combine heuristics with appropriate weights
    combined_heuristic = inverse_distance_heuristic + load_balance_heuristic

    # Adjust the heuristic values to ensure negative values for undesirable edges
    # and positive values for promising ones
    min_combined_heuristic = torch.min(combined_heuristic)
    combined_heuristic = combined_heuristic - min_combined_heuristic

    return combined_heuristic