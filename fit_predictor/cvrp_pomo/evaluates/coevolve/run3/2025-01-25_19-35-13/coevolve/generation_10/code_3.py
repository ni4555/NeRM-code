import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    # Initialize the heuristic matrix with a large negative value for all edges
    heuristics = torch.full(distance_matrix.shape, -float('inf'))
    # Calculate the heuristics based on normalized demands and distance
    heuristics[distance_matrix != 0] = -distance_matrix[distance_matrix != 0] * normalized_demands[distance_matrix != 0]
    return heuristics