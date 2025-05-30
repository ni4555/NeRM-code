import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity (sum of demands)
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Compute the negative of the demands for the heuristics calculation
    negative_demands = -normalized_demands

    # For each edge, compute the heuristic value as the negative demand multiplied by the distance
    # This encourages edges with lower distances and lower demands (i.e., less capacity needed)
    heuristics = negative_demands.view(-1, 1) * distance_matrix

    # Add a small positive constant to avoid log(0) and ensure all values are positive
    epsilon = 1e-8
    heuristics = torch.clamp(heuristics, min=epsilon)

    return heuristics