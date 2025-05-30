import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()
    demands_normalized = demands / vehicle_capacity

    # Calculate the negative of the demand as a heuristic for undesirable edges
    negative_demand = -demands_normalized

    # Calculate the distance-based heuristic
    distance_based_heuristic = distance_matrix

    # Combine the demand and distance-based heuristics
    combined_heuristic = negative_demand + distance_based_heuristic

    return combined_heuristic