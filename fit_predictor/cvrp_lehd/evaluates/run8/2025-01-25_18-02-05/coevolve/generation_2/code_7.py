import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    # Calculate the negative of the demand as a heuristic
    # Negative demand means a more promising edge (since we're minimizing)
    heuristics = -demands
    return heuristics