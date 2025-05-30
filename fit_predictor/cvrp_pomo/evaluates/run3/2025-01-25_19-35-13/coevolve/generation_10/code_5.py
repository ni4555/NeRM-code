import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    total_capacity = demands[0]  # Assuming the first node is the depot and its demand is the vehicle capacity
    normalized_demands = demands[1:] / total_capacity  # Normalize customer demands by the vehicle capacity
    max_demand = normalized_demands.max()
    # Use the normalized demand as a heuristic value for each edge
    heuristics = -normalized_demands.unsqueeze(0) + max_demand.unsqueeze(1)
    return heuristics