import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    demands = demands / total_capacity

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Compute the heuristic for each edge
    # The heuristic can be a combination of the demand and distance
    # Here we use a simple formula: -distance + demand
    heuristic_matrix = -distance_matrix + demands

    # Ensure the heuristic matrix is symmetrical
    # This is important as CVRP is an undirected problem
    symmetric_heuristic_matrix = (heuristic_matrix + heuristic_matrix.t()) / 2

    return symmetric_heuristic_matrix