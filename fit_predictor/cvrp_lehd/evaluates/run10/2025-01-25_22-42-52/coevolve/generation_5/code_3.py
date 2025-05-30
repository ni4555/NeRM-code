import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Compute the sum of the normalized demands for each edge
    edge_demands = torch.matmul(normalized_demands, distance_matrix)

    # Subtract edge demands from 1 to get the heuristic values
    # Negative values for undesirable edges, positive for promising ones
    heuristics = 1 - edge_demands

    return heuristics