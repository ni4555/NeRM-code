import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Compute the heuristic values based on the distance to the depot (first row) and demand
    # Subtract the demand from the distance to create a penalty for carrying more than the capacity
    heuristics = distance_matrix[0] - normalized_demands

    # Apply a small positive value to all edges to avoid zero values in the output tensor
    heuristics = torch.clamp(heuristics, min=1e-6)

    return heuristics