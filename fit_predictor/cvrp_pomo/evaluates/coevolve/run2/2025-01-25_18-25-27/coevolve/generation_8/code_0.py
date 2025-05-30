import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Calculate the heuristic values based on distance and normalized demand
    heuristic_matrix = distance_matrix * normalized_demands

    # Apply penalties for edges that lead to overloading
    for i in range(n):
        for j in range(n):
            if i != j:
                # Assuming that the depot is at index 0 and the demand at the depot is 0
                # The penalty is the sum of the demand at the customer node
                penalty = demands[j]
                heuristic_matrix[i, j] -= penalty

    return heuristic_matrix