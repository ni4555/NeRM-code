import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the heuristic as the negative demand (promising) plus the distance to the depot
    heuristics = -demands + distance_matrix[:, 0]  # Assuming the depot is at index 0
    return heuristics