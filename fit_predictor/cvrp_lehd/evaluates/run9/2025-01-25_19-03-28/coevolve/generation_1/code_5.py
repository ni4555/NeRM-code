import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix by the total vehicle capacity
    vehicle_capacity = demands.sum()
    normalized_distance_matrix = distance_matrix / vehicle_capacity

    # Calculate the heuristic values by considering the demands
    # Higher demands result in higher costs for including this edge
    heuristic_matrix = normalized_distance_matrix * demands.unsqueeze(1)

    # Add a penalty for high distance to discourage long routes
    # Here, we assume a simple quadratic penalty
    distance_penalty = distance_matrix ** 2
    heuristic_matrix = heuristic_matrix + distance_penalty

    # Subtract a large number for edges that go to the depot (0)
    # since we don't want to include edges to the depot
    depot_penalty = torch.ones_like(heuristic_matrix) * float('-inf')
    depot_penalty[torch.arange(heuristic_matrix.shape[0]), torch.arange(heuristic_matrix.shape[0])] = 0
    heuristic_matrix = heuristic_matrix + depot_penalty

    return heuristic_matrix