import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands[0]  # Assuming the first element of demands is the total vehicle capacity
    demands = demands[1:]  # Exclude the total vehicle capacity from demands vector

    # Calculate the load of each edge as the product of the distance and demand
    load_matrix = distance_matrix * demands

    # Calculate the normalized load for each edge
    normalized_load_matrix = load_matrix / vehicle_capacity

    # Calculate a penalty for edges that exceed the vehicle capacity
    penalties = (normalized_load_matrix - 1).clamp(min=0) * distance_matrix

    # Calculate the heuristic as the sum of the negative of the normalized load and penalties
    heuristic_matrix = -normalized_load_matrix + penalties

    # Add a small constant to ensure all values are finite and to avoid division by zero
    epsilon = 1e-8
    heuristic_matrix = heuristic_matrix / (heuristic_matrix + epsilon)

    return heuristic_matrix