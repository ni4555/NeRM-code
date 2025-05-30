import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demands_normalized = demands / total_capacity

    # Calculate the potential function using both distance and demand
    potential = (distance_matrix ** 2) * demands_normalized

    # Normalize the potential function to handle varying scales
    max_potential = torch.max(potential)
    epsilon = 1e-8  # Epsilon value to prevent division by zero
    normalized_potential = potential / (max_potential + epsilon)

    # Compute the heuristics based on the normalized potential
    heuristics = -normalized_potential

    return heuristics