import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the cost for each edge based on distance and demand
    edge_costs = distance_matrix * normalized_demands

    # Apply a simple heuristic: the lower the cost, the more promising the edge
    # We use a negative value for undesirable edges to ensure they are less likely to be selected
    heuristics = -edge_costs

    return heuristics