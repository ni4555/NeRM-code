import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the distance matrix is square
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be square")

    # Ensure demands are normalized
    if not torch.allclose(demands.sum(), torch.tensor(1.0)):
        raise ValueError("Demands vector must sum up to 1.0")

    # Initialize the heuristic matrix with high values for all edges
    heuristic_matrix = -torch.ones_like(distance_matrix)

    # Calculate the sum of demands from the depot to each customer
    demands_from_depot = demands * distance_matrix

    # Calculate the sum of demands from each customer to the depot
    demands_to_depot = demands * distance_matrix.transpose(0, 1)

    # Apply the heuristic based on demand flow and distance
    # Here, we use a simple heuristic that rewards shorter paths with higher demands
    heuristic_matrix = demands_from_depot + demands_to_depot

    return heuristic_matrix