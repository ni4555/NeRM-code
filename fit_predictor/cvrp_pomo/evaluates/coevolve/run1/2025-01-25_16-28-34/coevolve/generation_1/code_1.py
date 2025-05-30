import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the difference in demands between each pair of nodes
    demand_diff = demands[:, None] - demands[None, :]
    # Normalize demand difference by the total vehicle capacity
    normalized_demand_diff = demand_diff / torch.sum(demands)
    # Use the distance matrix to calculate a heuristic based on the demand difference
    # We use the square of the distance as a heuristic weight
    heuristic_matrix = distance_matrix ** 2 * normalized_demand_diff
    # For the diagonal elements, which represent the distance from a node to itself,
    # we set the heuristic to a very negative value to discourage including such edges
    diagonal_mask = torch.eye(n, dtype=torch.bool)
    heuristic_matrix[diagonal_mask] = -float('inf')
    return heuristic_matrix