import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the distance-demand heuristic
    # For each edge, the heuristic is the negative of the distance multiplied by the normalized demand
    # This encourages edges with lower distance and higher demand to be included in the solution
    heuristics = -distance_matrix * normalized_demands

    return heuristics