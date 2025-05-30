import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    depot_index = 0
    total_capacity = demands.sum()
    demands_normalized = demands / total_capacity

    # Calculate the heuristics for each edge based on the normalized demands
    heuristics = -distance_matrix ** 2 * demands_normalized

    # Add an additional term to encourage routes to include the depot at the end
    heuristics[depot_index, :] -= 1
    heuristics[:, depot_index] -= 1

    return heuristics