import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the heuristics matrix with high negative values for undesirable edges
    n = distance_matrix.shape[0]
    heuristics = torch.full((n, n), -float('inf'))

    # Calculate the initial heuristic values for each edge
    for i in range(1, n):  # skip the depot node
        for j in range(i + 1, n):  # only consider edges between customers
            # Calculate the heuristic value based on the normalized demand and distance
            heuristics[i, j] = heuristics[j, i] = normalized_demands[i] + normalized_demands[j] - distance_matrix[i, j]

    return heuristics