import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    demands_sum = demands.sum()
    normalized_demands = demands / demands_sum

    # Initialize heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the sum of distances for each edge
    sum_distances = torch.sum(distance_matrix, dim=1)

    # Calculate the heuristics based on the normalized demands and sum of distances
    heuristics = normalized_demands.unsqueeze(1) * sum_distances.unsqueeze(0)

    # Adjust heuristics for the depot node (0,0)
    heuristics[0, 0] = -float('inf')  # Depot to depot is not an edge

    return heuristics