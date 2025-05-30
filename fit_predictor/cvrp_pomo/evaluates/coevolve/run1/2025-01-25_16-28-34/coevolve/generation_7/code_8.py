import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()
    demands_normalized = demands / vehicle_capacity

    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the initial cost for each edge
    for i in range(n):
        for j in range(n):
            if i != j:
                # Edge cost is the demand normalized by the vehicle capacity
                edge_cost = demands_normalized[j]
                # Add the edge cost to the heuristics matrix
                heuristics[i, j] = edge_cost

    # Adjust the heuristics matrix to ensure it has the correct shape
    heuristics = heuristics.view_as(distance_matrix)

    return heuristics