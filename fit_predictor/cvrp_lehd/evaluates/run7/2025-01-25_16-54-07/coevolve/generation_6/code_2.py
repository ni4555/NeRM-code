import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demand vector by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Compute cumulative demand along each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Skip the depot node
                cumulative_demand = normalized_demands[i].sum() + normalized_demands[j]
                # Assign a heuristic value based on the cumulative demand
                heuristics[i, j] = -cumulative_demand

    return heuristics