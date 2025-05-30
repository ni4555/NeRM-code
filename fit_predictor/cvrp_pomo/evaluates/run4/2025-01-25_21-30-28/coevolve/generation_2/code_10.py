import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum() / n  # Normalize the vehicle capacity to the size of the matrix
    # Compute the sum of demands for each row and subtract from vehicle_capacity
    load_per_customer = vehicle_capacity - demands
    # The heuristics for an edge are based on the difference in demands
    heuristics = load_per_customer[1:] - load_per_customer[:-1] + distance_matrix[1:, :-1] - distance_matrix[:-1, 1:]
    return heuristics