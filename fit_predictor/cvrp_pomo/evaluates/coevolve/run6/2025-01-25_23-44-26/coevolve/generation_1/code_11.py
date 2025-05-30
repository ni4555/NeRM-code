import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the load on each edge as the sum of demands of both nodes connected by the edge
    load = demands[:, None] + demands[None, :] - 2 * demands * torch.eye(n, dtype=torch.float32)
    # Normalize the load by the total vehicle capacity
    load /= demands.sum()
    # Calculate the heuristics as the negative load for undesirable edges and zero for desirable edges
    heuristics = -torch.abs(load)
    return heuristics