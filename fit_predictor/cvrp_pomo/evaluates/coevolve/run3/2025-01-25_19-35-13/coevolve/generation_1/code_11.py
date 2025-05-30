import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristic value for each edge based on the normalized demand
    # and the inverse of the distance. This heuristic is inspired by the
    # savings heuristic for the Vehicle Routing Problem (VRP).
    heuristics = (normalized_demands.unsqueeze(1) + normalized_demands.unsqueeze(0) -
                  2 * distance_matrix / distance_matrix.sum())

    # Set the diagonal to a large negative value to avoid selecting the depot
    # as a customer in the heuristic evaluation.
    heuristics.diag().fill_(float('-inf'))

    return heuristics