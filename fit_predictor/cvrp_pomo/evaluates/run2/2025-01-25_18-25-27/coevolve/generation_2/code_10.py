import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the cumulative sum of demands from the depot
    cumulative_demands = torch.cumsum(demands, dim=0)
    # Calculate the cumulative sum of distances from the depot
    cumulative_distances = torch.cumsum(distance_matrix[0], dim=0)
    # Calculate the potential cost for each edge
    potential_costs = (cumulative_demands - demands) * cumulative_distances
    # Normalize the potential costs by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_potential_costs = potential_costs / total_capacity
    # Add a small constant to avoid division by zero
    return normalized_potential_costs + 1e-8