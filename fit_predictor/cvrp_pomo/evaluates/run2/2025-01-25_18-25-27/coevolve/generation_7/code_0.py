import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the initial heuristics based on normalized demands
    initial_heuristics = -normalized_demands.repeat_interleave(n, dim=0) - \
                         normalized_demands.repeat(n, 1)

    # Adjust heuristics based on distance (shorter distances should be more desirable)
    distance_diff = distance_matrix - distance_matrix.mean(dim=1, keepdim=True)
    distance_diff = torch.abs(distance_diff)
    adjusted_heuristics = initial_heuristics + distance_diff

    # Apply a penalty for edges leading to overloading
    overloading_penalty = torch.zeros_like(adjusted_heuristics)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                overloading_penalty[i, j] = 1e-3 * (adjusted_heuristics[i, j] < 0)
    adjusted_heuristics += overloading_penalty

    return adjusted_heuristics