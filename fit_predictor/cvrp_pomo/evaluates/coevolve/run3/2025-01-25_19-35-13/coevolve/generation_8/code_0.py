import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristics as a weighted sum of distance and demand
    # Here, we use a simple linear combination, but this can be adjusted as needed
    alpha = 0.5  # Weight for distance
    beta = 0.5   # Weight for demand
    heuristics = alpha * distance_matrix + beta * normalized_demands

    # Adjust heuristics to have negative values for undesirable edges and positive for promising ones
    # We do this by subtracting the maximum value from all elements to ensure non-negative heuristics
    heuristics -= heuristics.max()

    return heuristics