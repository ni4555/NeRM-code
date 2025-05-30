import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    demand_ratio = demands / demands.sum()  # Normalize demand
    return torch.sqrt(1 + (1 / distance_matrix)) - torch.sqrt(1 / (distance_matrix + demand_ratio))