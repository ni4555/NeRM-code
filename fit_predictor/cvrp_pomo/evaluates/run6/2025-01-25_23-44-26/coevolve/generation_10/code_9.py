import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that demands are normalized by the total vehicle capacity
    total_capacity = torch.sum(demands)
    normalized_demands = demands / total_capacity

    # Calculate the normalized distance matrix
    normalized_distance_matrix = distance_matrix / torch.max(distance_matrix)

    # Create a penalty function based on the distance to the next customer
    # Higher penalty for edges closer to the total vehicle capacity
    penalty_factor = torch.where(normalized_demands > 0.8, 1.5, 1.0)
    penalty = penalty_factor * normalized_distance_matrix

    # Incorporate the demand-driven heuristic: lower penalty for edges with lower demands
    demand_factor = torch.log(1 + normalized_demands)
    heuristic_value = penalty - demand_factor

    return heuristic_value