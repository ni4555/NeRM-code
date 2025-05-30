import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()
    demand_penalty = -1 * (distance_matrix > 0) * (distance_matrix * demands / vehicle_capacity)
    return demand_penalty