import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()

    # Calculate the negative distance heuristic
    negative_distance_heuristic = -distance_matrix

    # Calculate the demand heuristic
    demand_heuristic = demands / vehicle_capacity

    # Combine the heuristics, giving more weight to demand heuristic
    combined_heuristic = negative_distance_heuristic + 2 * demand_heuristic

    return combined_heuristic