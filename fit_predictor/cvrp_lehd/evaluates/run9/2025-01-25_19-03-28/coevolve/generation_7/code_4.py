import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demand between each pair of nodes
    demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)

    # Normalize the demand difference by the vehicle capacity (assumed to be 1 for normalization)
    normalized_demand_diff = demand_diff / demands.unsqueeze(1)

    # Calculate the heuristic as the sum of the normalized demand difference and the negative distance
    heuristics = normalized_demand_diff - distance_matrix

    # Ensure that the heuristics are negative for undesirable edges and positive for promising ones
    heuristics = torch.clamp(heuristics, min=-1e6, max=1e6)

    return heuristics