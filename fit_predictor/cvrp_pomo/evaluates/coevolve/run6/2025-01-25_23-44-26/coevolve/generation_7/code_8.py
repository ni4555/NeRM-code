import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Compute the inverse distance heuristic
    # The heuristic is based on the inverse of the distance, so we take the reciprocal of the distance matrix
    # and subtract it from 1 to get negative values for closer customers
    inverse_distance = 1 - (distance_matrix / distance_matrix.max())

    # Demand-penalty mechanism: penalize edges with high demand
    demand_penalty = normalized_demands * distance_matrix

    # Combine the inverse distance heuristic with the demand penalty
    combined_heuristic = inverse_distance - demand_penalty

    # Ensure that all values are within a reasonable range (e.g., between -1 and 1)
    combined_heuristic = torch.clamp(combined_heuristic, min=-1, max=1)

    return combined_heuristic