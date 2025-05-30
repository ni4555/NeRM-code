import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize demands to the range [0, 1]
    demands_normalized = demands / demands.sum()
    # Compute the "promise" of each edge based on the normalized demand
    promise_matrix = demands_normalized.unsqueeze(1) * distance_matrix.unsqueeze(0)
    # The heuristic for each edge is negative if it's a depot-to-depot edge and positive otherwise
    heuristic_matrix = torch.where(distance_matrix.eq(0), -promise_matrix, promise_matrix)
    return heuristic_matrix