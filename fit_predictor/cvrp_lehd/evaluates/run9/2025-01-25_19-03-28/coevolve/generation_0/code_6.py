import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristic for each edge as the negative of the normalized demand
    # multiplied by the distance squared (to simulate a heuristic where shorter distances
    # with lower demands are more promising).
    # Note: This heuristic is a simple example and can be replaced with more sophisticated methods.
    heuristic_matrix = -torch.mul(normalized_demands, distance_matrix ** 2)

    return heuristic_matrix