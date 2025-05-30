import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    if total_capacity == 0:
        raise ValueError("Total vehicle capacity must be greater than zero.")
    normalized_demands = demands / total_capacity

    # Calculate a simple heuristic based on distance and normalized demand
    # Negative values indicate undesirable edges, positive values indicate promising ones
    heuristics = distance_matrix - normalized_demands

    # Adjust heuristics to ensure all values are non-negative (promising)
    heuristics = torch.clamp(heuristics, min=0)

    return heuristics