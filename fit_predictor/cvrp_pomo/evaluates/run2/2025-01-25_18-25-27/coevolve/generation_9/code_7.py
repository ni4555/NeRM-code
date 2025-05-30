import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the cost of visiting each customer from the depot
    visit_cost = distance_matrix[0] * normalized_demands

    # Calculate the heuristic value for each edge
    heuristics = -visit_cost

    # Apply normalization and constraint handling to adjust the heuristic values
    # This is a simple normalization step, more complex normalization can be used
    heuristics = (heuristics - heuristics.min()) / (heuristics.max() - heuristics.min())

    return heuristics