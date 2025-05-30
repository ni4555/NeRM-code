import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the heuristic values based on the normalized demands
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # For each edge, calculate the heuristic value as the negative of the demand ratio
                # This encourages the selection of edges with lower demand ratio (promising edges)
                heuristics[i, j] = -normalized_demands[j]

    # Adjust the heuristic values to ensure they are negative for undesirable edges
    # and positive for promising ones
    heuristics[distance_matrix == 0] = 0  # Set the depot edges to 0
    heuristics[distance_matrix != 0] = heuristics[distance_matrix != 0].clamp(min=0)

    return heuristics