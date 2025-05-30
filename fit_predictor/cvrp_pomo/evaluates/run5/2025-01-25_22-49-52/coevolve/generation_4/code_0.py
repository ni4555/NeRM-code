import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Normalize the distance matrix
    distance_matrix = distance_matrix / distance_matrix.max()

    # Calculate potential values for explicit depot handling
    depot_potential = distance_matrix.sum(dim=1) - distance_matrix[:, 0]

    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the potential value for each edge
    for i in range(1, len(distance_matrix)):
        for j in range(1, len(distance_matrix)):
            # Calculate the potential value for the edge (i, j)
            edge_potential = (depot_potential[i] + depot_potential[j] - distance_matrix[i, j] * 2) * normalized_demands[i]
            heuristics[i, j] = edge_potential

    return heuristics