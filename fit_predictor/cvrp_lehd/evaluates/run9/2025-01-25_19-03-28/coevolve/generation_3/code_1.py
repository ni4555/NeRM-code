import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristic values for each edge
    # The heuristic will be a weighted sum of inverse distance and normalized demand
    # We use a negative distance to discourage longer routes and a positive demand to encourage high-demand routes
    heuristic_values = -distance_matrix + normalized_demands

    # We can also add a small constant to avoid division by zero in the next step
    epsilon = 1e-6
    heuristic_values = heuristic_values + epsilon

    # Normalize the heuristic values to ensure they are within a certain range
    # This step is optional but can help improve the performance of some optimization algorithms
    max_value = heuristic_values.max()
    min_value = heuristic_values.min()
    normalized_heuristics = (heuristic_values - min_value) / (max_value - min_value)

    return normalized_heuristics