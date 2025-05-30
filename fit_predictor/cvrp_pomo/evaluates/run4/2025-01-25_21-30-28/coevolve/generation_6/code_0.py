import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse distance for each edge
    inverse_distance = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero

    # Combine the inverse distance with the normalized demands
    heuristics = inverse_distance * normalized_demands

    # Return the heuristics matrix
    return heuristics