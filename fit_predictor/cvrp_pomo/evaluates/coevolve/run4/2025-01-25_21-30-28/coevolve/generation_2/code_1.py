import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Compute the potential cost for each edge
    # Here we use a simple heuristic: the higher the demand, the more promising the edge
    # This is a basic example and can be replaced with more sophisticated heuristics
    heuristics = normalized_demands.unsqueeze(1) - normalized_demands.unsqueeze(0)

    # To ensure the diagonal is zero (self-loops are not desirable)
    heuristics = heuristics.masked_fill(heuristics == 0, float('-inf'))

    return heuristics