import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse distance heuristic
    inv_distance = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero

    # Calculate the Normalization heuristic
    normalization = normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)

    # Combine the heuristics
    combined_heuristics = inv_distance * normalization

    # Apply a weight to the heuristics to emphasize the inverse distance
    weight = 0.5  # This weight can be adjusted for different problem instances
    heuristics = weight * combined_heuristics + (1 - weight) * inv_distance

    return heuristics