import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand mask
    cumulative_demand = demands.cumsum(dim=0)

    # Normalize the cumulative demand by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demand = cumulative_demand / total_capacity

    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Compute the edge feasibility mask
    edge_capacity_mask = distance_matrix < normalized_demand

    # Apply capacity-based prioritization
    heuristics[distance_matrix < normalized_demand] = -1
    heuristics[distance_matrix == normalized_demand] = 0

    return heuristics