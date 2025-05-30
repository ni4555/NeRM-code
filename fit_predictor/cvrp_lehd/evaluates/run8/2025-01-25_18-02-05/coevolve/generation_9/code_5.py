import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    demands_normalized = demands / total_capacity

    # Compute the heuristic values
    # The heuristic could be a combination of demand and distance factors.
    # Here we use a simple example where we weigh the demand and distance.
    # The negative sign ensures that lower heuristic values are more promising.
    # This is just a placeholder; actual heuristic should be tailored to the problem specifics.
    heuristic_matrix = -demands_normalized.unsqueeze(1) * distance_matrix.unsqueeze(0) + 1

    return heuristic_matrix