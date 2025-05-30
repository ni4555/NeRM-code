import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized by the total vehicle capacity
    vehicle_capacity = demands.sum()
    demands = demands / vehicle_capacity

    # Calculate the heuristic values using the formula:
    # heuristic = distance + demand_weight * (1 - load_factor)
    # where load_factor = demand / vehicle_capacity
    load_factor = demands / vehicle_capacity
    heuristic = distance_matrix + demands * (1 - load_factor)

    return heuristic