import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the load difference between each edge
    load_diff = distance_matrix * (demands[distance_matrix > 0] - demands[distance_matrix <= 0])

    # Normalize the load difference by the vehicle capacity
    vehicle_capacity = 1.0  # Assuming the total vehicle capacity is 1 for normalization
    normalized_load_diff = load_diff / vehicle_capacity

    # Calculate the distance cost for each edge
    distance_cost = distance_matrix

    # Combine load difference and distance cost into a single heuristic value
    heuristic_values = normalized_load_diff - distance_cost

    return heuristic_values