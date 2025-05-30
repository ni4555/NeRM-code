import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum distance from the depot to any other node
    max_distance = torch.max(distance_matrix[0, 1:], dim=0).values.unsqueeze(0)

    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the potential cost for each edge based on the normalized demand
    demand_cost = (distance_matrix - max_distance) * normalized_demands

    # Use a simple heuristic: edges with higher demand cost are more promising
    # In this case, we subtract the demand cost since we want negative values for undesirable edges
    heuristics = -demand_cost

    return heuristics