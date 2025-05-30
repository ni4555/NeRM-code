import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to be between 0 and 1
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Normalize distances using the demands to create a weighted distance matrix
    # The idea is to weight distances by the demands to prioritize edges with higher demands
    weighted_distance_matrix = distance_matrix * normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)

    # The heuristic is the negative of the weighted distance, as we want to minimize the sum of weighted distances
    heuristics = -weighted_distance_matrix

    return heuristics