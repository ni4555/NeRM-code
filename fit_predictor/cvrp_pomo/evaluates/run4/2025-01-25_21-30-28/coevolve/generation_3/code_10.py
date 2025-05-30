import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristic values based on edge weights and demand
    # Here we use a simple heuristic based on the ratio of demand to distance
    # This can be replaced with more complex heuristics as needed
    heuristic_matrix = -torch.abs(distance_matrix) / (normalized_demands[:, None] + 1e-8)

    # Normalize the heuristic matrix to ensure all values are within the same scale
    # This helps in the genetic algorithm phase to maintain diversity in the population
    min_val = heuristic_matrix.min()
    max_val = heuristic_matrix.max()
    normalized_heuristic_matrix = (heuristic_matrix - min_val) / (max_val - min_val)

    return normalized_heuristic_matrix