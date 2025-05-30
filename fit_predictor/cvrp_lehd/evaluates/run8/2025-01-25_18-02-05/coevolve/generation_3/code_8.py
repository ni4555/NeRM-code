import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix, dtype=torch.float32)
    # Compute the relative demands for all customers
    relative_demands = demands / demands.sum()
    # Compute the heuristics for each edge
    for i in range(n):
        for j in range(n):
            if i != j:
                # The heuristic for each edge is based on the distance and the demand
                heuristics[i, j] = -distance_matrix[i, j] + relative_demands[j]
    return heuristics