import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demand_normalized = demands / total_capacity

    # Calculate the heuristic values based on the normalized demands
    # We use a simple heuristic where we consider edges with lower distance and higher demand as more promising
    # The heuristic is negative for undesirable edges and positive for promising ones
    heuristics = -distance_matrix + demand_normalized

    # Normalize the heuristics to ensure they are within a certain range
    heuristics = torch.clamp(heuristics, min=-1.0, max=1.0)

    return heuristics