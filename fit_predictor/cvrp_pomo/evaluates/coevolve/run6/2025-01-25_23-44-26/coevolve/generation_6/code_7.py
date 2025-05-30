import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize customer demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Inverse Distance Heuristic (IDH) - assign customers based on the reciprocal of their distance
    # Ensure that the distance matrix is squared to avoid double counting
    distance_matrix_squared = distance_matrix ** 2
    idh_scores = 1 / distance_matrix_squared

    # Demand penalty function - increase cost for high-demand customers near capacity limits
    # Calculate the sum of demands for each vehicle considering the depot
    vehicle_capacities = torch.clamp(demands.cumsum() - demands[0], min=0)
    demand_penalty = (vehicle_capacities / total_capacity) ** 2

    # Combine IDH scores with demand penalty to get the heuristic scores
    heuristic_scores = idh_scores - demand_penalty

    # Ensure that the heuristic scores are negative for undesirable edges and positive for promising ones
    # This is done by adding a large constant to the negative values and subtracting it from the positive values
    large_constant = 1e6
    heuristic_scores = torch.where(heuristic_scores < 0, large_constant + heuristic_scores, large_constant - heuristic_scores)

    return heuristic_scores