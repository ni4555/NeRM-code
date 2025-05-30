import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming the distance matrix is symmetric and the depot node is indexed by 0
    n = distance_matrix.shape[0]
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    # Normalize the demands
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum
    # Compute the heuristic for each edge
    for i in range(n):
        for j in range(1, n):
            if demand_sum[i] < 1.0:
                # Compute the heuristic as the negative of the demand
                # multiplied by the distance
                heuristic = -normalized_demands[i] * distance_matrix[i, j]
                # Store the heuristic in the matrix
                heuristic_matrix[i, j] = heuristic
    return heuristic_matrix