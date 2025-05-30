import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity (assuming total capacity is 1 for simplicity)
    total_capacity = 1.0
    normalized_demands = demands / total_capacity

    # Calculate path potential based on distance and demand
    # The heuristic function is designed to favor shorter distances and lower demands
    path_potential = distance_matrix * (1 - normalized_demands)

    # Normalize path potential for consistent scaling
    # We use a simple min-max normalization here
    min_potential = path_potential.min()
    max_potential = path_potential.max()
    normalized_potential = (path_potential - min_potential) / (max_potential - min_potential)

    # Invert the normalized potential to give negative values to undesirable edges and positive to promising ones
    heuristics = -normalized_potential

    return heuristics