import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand difference matrix
    normalized_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    # Ensure non-positive values (undesirable edges) are set to -inf
    undesirable_edges = torch.where(normalized_diff <= 0, -torch.inf, 0)
    # Ensure positive values (promising edges) are set to the absolute difference
    promising_edges = torch.where(normalized_diff > 0, normalized_diff.abs(), 0)
    # Combine the desirable and undesirable edges, prioritizing promising ones
    return promising_edges - undesirable_edges