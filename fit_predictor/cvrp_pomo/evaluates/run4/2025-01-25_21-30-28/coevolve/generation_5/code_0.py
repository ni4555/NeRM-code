import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse distance for each edge
    inverse_distance = 1 / distance_matrix

    # Normalize the inverse distance by the sum of the normalized demands
    normalized_inverse_distance = inverse_distance / normalized_demands.sum()

    # Subtract the normalized demands from the normalized inverse distance
    # to create a heuristic matrix
    heuristic_matrix = normalized_inverse_distance - normalized_demands

    return heuristic_matrix