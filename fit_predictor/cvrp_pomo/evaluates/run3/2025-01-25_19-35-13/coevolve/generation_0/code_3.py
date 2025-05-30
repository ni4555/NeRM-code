import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the distance matrix is of shape (n, n) and the demands vector is of shape (n,)
    n = distance_matrix.shape[0]
    assert distance_matrix.shape == (n, n), "Distance matrix must be of shape (n, n)"
    assert demands.shape == (n,), "Demands vector must be of shape (n,)"

    # Initialize the heuristic matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the total demand
    total_demand = demands.sum()

    # Iterate over all pairs of nodes (i, j) where i is not equal to j
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate the heuristic value for the edge (i, j)
                # This is a simple heuristic where we consider the negative demand at node j
                # and the distance from node i to node j.
                heuristics[i, j] = -demands[j] + distance_matrix[i, j]

    # Normalize the heuristic values by the total demand
    heuristics /= total_demand

    return heuristics