import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristic values based on the normalized demands
    # We will use a simple heuristic: the negative of the demand (lower demand is better)
    # This is a demand relaxation approach, which is a common heuristic for capacitated VRP
    heuristic_matrix = -normalized_demands

    # Incorporate dynamic window approach by adding a penalty for longer distances
    # This can be done by subtracting the distance matrix from a constant (e.g., max distance)
    max_distance = distance_matrix.max().item()
    heuristic_matrix = heuristic_matrix - distance_matrix

    # Incorporate path decomposition by adding a penalty for edges that exceed a certain threshold
    # This is a simplistic approach, and more sophisticated methods can be used depending on the problem
    threshold = 0.5 * max_distance
    heuristic_matrix[distance_matrix > threshold] += distance_matrix[distance_matrix > threshold]

    return heuristic_matrix