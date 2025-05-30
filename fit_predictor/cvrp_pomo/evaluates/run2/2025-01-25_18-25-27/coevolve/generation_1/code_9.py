import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize a tensor with the same shape as distance_matrix filled with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the negative demand for each edge as a heuristic
    # This assumes that shorter distances are better and demand should be a factor
    heuristics = -distance_matrix * normalized_demands.expand(n, n)

    # Add a small constant to avoid division by zero
    epsilon = 1e-8
    heuristics = heuristics / (torch.sum(normalized_demands, dim=0, keepdim=True) + epsilon)

    return heuristics