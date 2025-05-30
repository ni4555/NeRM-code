import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands vector to the range [0, 1]
    demands_normalized = demands / demands.sum()

    # Calculate the maximum demand as a baseline for comparison
    max_demand = demands.max()

    # Compute the heuristics as the product of normalized demands and distance matrix
    heuristics = demands_normalized * distance_matrix

    # Subtract the maximum demand from the heuristics to make the values negative for undesirable edges
    heuristics -= max_demand

    # Add a small constant to avoid zero values which can cause numerical instability
    heuristics += 1e-10

    return heuristics