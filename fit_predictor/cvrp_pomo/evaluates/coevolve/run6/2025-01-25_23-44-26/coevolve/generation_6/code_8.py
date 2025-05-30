import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize customer demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse distance heuristic (IDH) using reciprocal distance
    idh_values = 1.0 / distance_matrix

    # Combine IDH with normalized demand to create initial heuristics
    combined_heuristics = idh_values * normalized_demands

    # Calculate the demand penalty function
    demand_penalty = demands / demands.clamp(min=1e-8)  # Avoid division by zero

    # Increase the cost of assigning high-demand customers to vehicles near their capacity limits
    combined_heuristics = combined_heuristics + demand_penalty

    return combined_heuristics