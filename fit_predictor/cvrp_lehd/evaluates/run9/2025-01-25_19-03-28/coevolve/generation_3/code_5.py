import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse of the demand for balancing the load
    inverse_demand = 1 / (normalized_demands + 1e-8)  # Adding a small constant to avoid division by zero

    # Create a heuristic matrix based on distance, demand, and inverse demand
    # Negative values for undesirable edges, positive for promising ones
    # Using a weighted sum of distance, inverse demand, and demand (normalized)
    # The weights can be adjusted to favor certain criteria
    weight_distance = 1.0
    weight_inverse_demand = 1.0
    weight_demand = 1.0

    # Vectorized computation of the heuristic values
    heuristic_matrix = weight_distance * distance_matrix \
                      - weight_inverse_demand * inverse_demand \
                      + weight_demand * normalized_demands

    return heuristic_matrix