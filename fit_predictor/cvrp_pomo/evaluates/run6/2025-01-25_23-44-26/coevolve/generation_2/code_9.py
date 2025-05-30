import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the sum of normalized demands for each edge
    edge_demands = torch.matmul(normalized_demands.unsqueeze(1), normalized_demands.unsqueeze(0))

    # Calculate the heuristic values based on the edge demands
    # We use a simple heuristic where edges with higher demands are more promising
    heuristics = -edge_demands

    return heuristics