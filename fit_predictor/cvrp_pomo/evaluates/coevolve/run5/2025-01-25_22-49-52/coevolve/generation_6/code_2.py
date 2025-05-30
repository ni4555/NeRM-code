import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized
    total_capacity = demands.sum()
    demands = demands / total_capacity

    # Calculate the potential value for each edge
    # Potential value = -distance (to discourage longer distances) + demand (to encourage high-demand edges)
    potential_value = -distance_matrix + demands[:, None] * demands

    # Normalize potential values to have a meaningful comparison
    max_potential = potential_value.max()
    min_potential = potential_value.min()
    normalized_potential = (potential_value - min_potential) / (max_potential - min_potential)

    # Threshold to determine if an edge is promising or not
    threshold = 0.5

    # Create a mask for promising edges (normalized potential > threshold)
    promising_mask = normalized_potential > threshold

    # Create a tensor of the same shape as distance_matrix filled with negative values
    heuristic_matrix = -torch.ones_like(distance_matrix)

    # Replace promising edges with positive values
    heuristic_matrix[promising_mask] = normalized_potential[promising_mask]

    return heuristic_matrix