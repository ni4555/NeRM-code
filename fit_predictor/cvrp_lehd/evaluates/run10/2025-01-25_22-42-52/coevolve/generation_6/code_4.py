import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    # Normalize the customer demands
    normalized_demands = demands / total_capacity

    # Calculate the heuristics as the negative of the distances multiplied by the demand
    # to encourage including edges that serve high-demand customers.
    heuristics = -distance_matrix * normalized_demands

    return heuristics