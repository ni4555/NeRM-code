import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristic values based on normalized demands
    # We will use a simple heuristic where the heuristic value is inversely proportional to the demand
    # This means higher demand nodes will have lower heuristic values (undesirable edges)
    # and lower demand nodes will have higher heuristic values (promising edges)
    heuristic_values = 1 / (normalized_demands + 1e-8)  # Adding a small value to avoid division by zero

    # The distance_matrix is used to scale the heuristic values by the distance
    # We will assume that the distance_matrix is pre-normalized to [0, 1] range
    # and we will use it to weight the heuristic values
    heuristic_values = heuristic_values * distance_matrix

    return heuristic_values