import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Check if the input tensors are on the same device
    if not (distance_matrix.is_cuda == demands.is_cuda):
        raise ValueError("Distance matrix and demands tensor must be on the same device.")

    # Ensure the distance matrix is square and the demands tensor is 1-dimensional
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be square.")
    if demands.ndim != 1 or demands.shape[0] != distance_matrix.shape[0]:
        raise ValueError("Demands tensor must be 1-dimensional and match the number of nodes in the distance matrix.")

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Calculate the total demand
    total_demand = demands.sum()

    # Normalize demands by the total capacity
    normalized_demands = demands / total_demand

    # Calculate the heuristic values for each edge
    # Negative values for undesirable edges (high demand or high distance)
    # Positive values for promising edges (low demand or short distance)
    heuristic_matrix = -normalized_demands.unsqueeze(1) - normalized_demands.unsqueeze(0)
    heuristic_matrix += distance_matrix

    # The heuristic matrix now contains negative values for undesirable edges
    # and positive values for promising ones

    return heuristic_matrix