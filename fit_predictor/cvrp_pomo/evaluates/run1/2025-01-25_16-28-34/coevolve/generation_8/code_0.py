import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demands from the normalized demand
    diff_demands = demands - demands.mean()

    # Calculate the load factor for each edge
    load_factor = torch.clamp(diff_demands / demands.mean(), min=-1, max=1)

    # Calculate the heuristic value based on distance and load factor
    heuristic_values = distance_matrix * load_factor

    return heuristic_values