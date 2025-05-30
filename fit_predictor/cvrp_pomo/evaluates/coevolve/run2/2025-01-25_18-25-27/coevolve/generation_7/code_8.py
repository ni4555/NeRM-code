import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the heuristic matrix with high negative values
    heuristic_matrix = -torch.ones_like(distance_matrix)

    # Incorporate customer demands into heuristic matrix
    demand_penalty = 2 * (1 - normalized_demands)
    heuristic_matrix += demand_penalty * distance_matrix

    # Incorporate some additional heuristics if needed (e.g., distance-based)
    # For example, a simple distance-based heuristic could be:
    # heuristic_matrix += -distance_matrix

    return heuristic_matrix