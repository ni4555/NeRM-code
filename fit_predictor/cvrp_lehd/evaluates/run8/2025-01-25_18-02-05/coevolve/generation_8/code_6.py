import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the negative of the normalized demands to penalize heavy demands
    demand_penalty = -normalized_demands

    # Create a distance penalty matrix that increases with distance
    distance_penalty = distance_matrix.clone()
    # Avoiding division by zero by adding a small epsilon
    epsilon = 1e-6
    distance_penalty = torch.clamp(distance_penalty, min=epsilon)

    # Calculate the heuristic values as the sum of demand penalty and distance penalty
    heuristic_matrix = demand_penalty + distance_penalty

    return heuristic_matrix