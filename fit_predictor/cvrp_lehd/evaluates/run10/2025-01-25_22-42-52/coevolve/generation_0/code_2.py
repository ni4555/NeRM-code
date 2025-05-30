import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Create a mask for edges where the demand is above the capacity of a single vehicle
    high_demand_mask = normalized_demands > 1.0

    # Create a mask for edges where the demand is below the capacity of a single vehicle
    low_demand_mask = normalized_demands <= 1.0

    # For edges with high demand, assign a negative heuristic value to discourage selection
    high_demand_heuristics = -torch.ones_like(distance_matrix) * (high_demand_mask * distance_matrix)

    # For edges with low demand, calculate the heuristic value as the negative distance
    # This assumes that the lower the distance, the more promising the edge is
    low_demand_heuristics = -distance_matrix * low_demand_mask

    # Combine the two masks to create the final heuristics matrix
    heuristics_matrix = high_demand_heuristics + low_demand_heuristics

    return heuristics_matrix