import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()

    # Normalize the demands to the total vehicle capacity
    normalized_demands = demands / total_capacity

    # Calculate the heuristics based on normalized demands
    # Using the following heuristic:
    # heuristic = - (distance^2 * demand)
    # This heuristic encourages shorter distances and lower demands
    heuristics = - (distance_matrix ** 2 * normalized_demands)

    # Ensure that the heuristics matrix has the same shape as the distance matrix
    assert heuristics.shape == distance_matrix.shape, "Heuristics matrix shape does not match distance matrix shape."

    return heuristics