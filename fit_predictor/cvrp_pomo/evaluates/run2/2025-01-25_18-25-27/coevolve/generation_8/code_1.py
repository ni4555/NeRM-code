import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize demands by the total vehicle capacity (sum of all demands)
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Calculate the heuristic values for each edge
    # The heuristic is based on the normalized demand of the destination node
    for i in range(n):
        for j in range(n):
            if i != j:
                # The heuristic value is negative for the depot node to encourage leaving it
                if i == 0:
                    heuristic_matrix[i, j] = -normalized_demands[j]
                else:
                    # The heuristic value is positive for customer nodes
                    heuristic_matrix[i, j] = normalized_demands[j]

    return heuristic_matrix