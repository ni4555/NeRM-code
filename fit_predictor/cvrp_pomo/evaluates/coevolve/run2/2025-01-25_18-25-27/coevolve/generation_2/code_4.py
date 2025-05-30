import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the demand to capacity ratio for each customer
    demand_ratio = demands / demands.sum()
    # Normalize the distance matrix
    normalized_distance = distance_matrix / distance_matrix.sum()
    # Calculate the heuristic value for each edge
    heuristic_matrix = (1 - demand_ratio) * normalized_distance
    return heuristic_matrix