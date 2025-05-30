import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()

    # Normalize the demands to the total vehicle capacity
    normalized_demands = demands / total_capacity

    # Calculate the inverse of the distance matrix
    inv_distance_matrix = 1 / (distance_matrix + 1e-8)  # Adding a small value to avoid division by zero

    # Apply the Normalization heuristic
    normalization_heuristic = normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)

    # Apply the Inverse Distance heuristic
    inverse_distance_heuristic = -inv_distance_matrix

    # Combine the heuristics
    combined_heuristic = normalization_heuristic + inverse_distance_heuristic

    return combined_heuristic