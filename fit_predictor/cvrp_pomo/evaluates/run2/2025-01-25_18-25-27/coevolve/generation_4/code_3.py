import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Calculate the heuristic for each edge
    for i in range(n):
        for j in range(1, n):  # Skip the depot node
            # The heuristic is based on the demand of the customer and the distance
            # We use negative values for undesirable edges to encourage the evolutionary algorithm to avoid them
            heuristic_matrix[i, j] = -normalized_demands[j] - distance_matrix[i, j]

    return heuristic_matrix