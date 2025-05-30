import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristic values
    # We use the formula: heuristic = distance - demand
    # Negative values for undesirable edges (high demand, low distance)
    # Positive values for promising edges (low demand, high distance)
    heuristics = distance_matrix - normalized_demands.unsqueeze(1)

    return heuristics