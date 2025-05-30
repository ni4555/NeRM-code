import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized
    demands = demands / demands.sum()

    # Initialize the savings matrix with zeros
    savings_matrix = torch.zeros_like(distance_matrix)

    # Compute the savings for each customer
    # Savings are computed as the distance from the depot to the customer plus
    # the distance from the customer to the end of the route minus the distance
    # from the depot to the end of the route
    for i in range(1, len(demands)):
        savings_matrix[0, i] = distance_matrix[0, i] + distance_matrix[i, 0] - 2 * distance_matrix[i, 0]
        for j in range(i + 1, len(demands)):
            savings_matrix[i, j] = distance_matrix[i, j] + distance_matrix[j, 0] + distance_matrix[i, 0] - 2 * distance_matrix[i, 0]
            savings_matrix[j, i] = savings_matrix[i, j]  # Since the matrix is symmetric

    # Normalize the savings matrix by the vehicle capacity (which is 1 after normalization)
    savings_matrix = savings_matrix * demands.unsqueeze(0)

    return savings_matrix