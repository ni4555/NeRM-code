import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the negative distance from each node to the depot (including the depot itself)
    # and the negative normalized demand from each node to the depot.
    # We want to discourage paths with high demand or long distance.
    negative_distances = -distance_matrix
    negative_normalized_demands = -normalized_demands

    # The heuristic value for each edge is the sum of the negative distance and the negative normalized demand.
    # Negative values indicate undesirable edges (high distance or high demand), and positive values indicate promising edges.
    heuristics = negative_distances + negative_normalized_demands

    return heuristics