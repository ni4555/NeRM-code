import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Normalize distance matrix to create a comparable scale
    normalized_distance_matrix = distance_matrix / distance_matrix.max()

    # Calculate the heuristic value as a combination of normalized demand and distance
    heuristic_values = normalized_demands.unsqueeze(1) * normalized_distance_matrix.unsqueeze(0)

    # Subtract the heuristic value from the distance to create negative values for promising edges
    heuristics = distance_matrix - heuristic_values

    return heuristics