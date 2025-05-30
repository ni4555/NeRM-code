import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Assuming demands are normalized by the total vehicle capacity
    # Initialize a matrix to store the heuristics for each edge
    heuristics = torch.zeros_like(distance_matrix)

    # Loop through all pairs of customers (i, j)
    for i in range(n):
        for j in range(n):
            if i != j:  # Avoid the depot node
                # Calculate the total savings of choosing edge (i, j)
                savings = demands[i] + demands[j] - distance_matrix[i, j]
                heuristics[i, j] = savings

    return heuristics