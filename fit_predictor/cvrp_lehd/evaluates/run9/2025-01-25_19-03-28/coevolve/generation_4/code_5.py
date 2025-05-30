import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Compute the cumulative sum of demands from the end to the start
    cumulative_demands = torch.cumsum(demands, dim=0)[::-1]
    # Calculate the "excess" demand at each node, considering the total vehicle capacity
    excess_demands = demands - cumulative_demands[1:]
    # Compute the "heuristic" values as the negative of the excess demand
    # to penalize edges that would result in an overflow of vehicle capacity
    heuristic_matrix = -excess_demands
    # Normalize the heuristic matrix to ensure all values are within the desired range
    heuristic_matrix = (heuristic_matrix - heuristic_matrix.min()) / (heuristic_matrix.max() - heuristic_matrix.min())
    # Replace the diagonal with a large negative value to avoid choosing the depot node
    diag_indices = torch.arange(heuristic_matrix.size(0))
    heuristic_matrix[diag_indices, diag_indices] = -torch.inf
    return heuristic_matrix