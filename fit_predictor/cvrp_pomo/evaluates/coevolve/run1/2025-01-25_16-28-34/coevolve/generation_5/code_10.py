import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the potential cost of each edge
    # This is a simple heuristic that assumes a higher cost for edges that lead to overloading
    edge_costs = (distance_matrix ** 2) * (1 + torch.abs(normalized_demands))

    # Introduce a negative bias for edges leading to the depot (except for the initial move)
    edge_costs[torch.arange(edge_costs.shape[0]), 0] *= -1
    edge_costs[0, torch.arange(edge_costs.shape[1])] *= -1

    return edge_costs