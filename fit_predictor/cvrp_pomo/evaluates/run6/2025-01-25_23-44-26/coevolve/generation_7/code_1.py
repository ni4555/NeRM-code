import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demand vector by the sum of demands to ensure each demand is relative to the total capacity
    total_demand = demands.sum().item()
    normalized_demands = demands / total_demand

    # Calculate the inverse distance heuristic
    # Assuming the distance matrix is already precomputed and contains positive values
    min_distance = distance_matrix.min(dim=1, keepdim=True)[0]
    max_distance = distance_matrix.max(dim=1, keepdim=True)[0]
    inverse_distance = 1 / (min_distance + max_distance)

    # Integrate demand-penalty mechanism to deter overloading vehicles
    demand_penalty = -normalized_demands * 1000  # Example penalty factor

    # Combine the inverse distance and demand-penalty
    combined_heuristic = inverse_distance + demand_penalty

    return combined_heuristic