import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Calculate the total vehicle capacity
    total_capacity = torch.sum(demands)

    # Normalize customer demands by the total vehicle capacity
    normalized_demands = demands / total_capacity

    # Calculate the heuristic value for each edge based on normalized demand and distance
    # We use a simple heuristic where we penalize edges with higher distance and higher demand
    heuristic_matrix = -distance_matrix * normalized_demands

    return heuristic_matrix