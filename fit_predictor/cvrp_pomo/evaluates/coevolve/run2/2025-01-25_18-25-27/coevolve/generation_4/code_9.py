import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize a matrix with zeros of the same shape as the distance matrix
    heuristics_matrix = torch.zeros_like(distance_matrix)

    # Calculate the total vehicle capacity
    total_capacity = demands.sum()

    # Normalize the demands to the total vehicle capacity
    normalized_demands = demands / total_capacity

    # Calculate the normalized demand for each customer (excluding the depot)
    normalized_demands = normalized_demands[1:]

    # Calculate the heuristic values for each edge
    # For each edge (i, j) with i not equal to j, calculate the heuristic
    # The heuristic is a function of the distance and normalized demand
    # Here we use a simple heuristic as an example: -distance + normalized_demand
    # The negative sign makes shorter distances more desirable
    for i in range(1, n):  # start from 1 to skip the depot
        for j in range(1, n):  # start from 1 to skip the depot
            heuristics_matrix[i, j] = -distance_matrix[i, j] + normalized_demands[i]

    # Return the heuristics matrix
    return heuristics_matrix