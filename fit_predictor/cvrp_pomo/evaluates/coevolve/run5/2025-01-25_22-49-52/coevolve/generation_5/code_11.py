import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demands_normalized = demands / total_capacity

    # Path potential heuristic: combine distance and demand
    distance_potential = distance_matrix
    demand_potential = demands_normalized.unsqueeze(1) * demands_normalized.unsqueeze(0)

    # Normalize the potential to ensure consistent scaling
    max_potential = torch.max(torch.abs(distance_potential) + torch.abs(demand_potential))
    normalized_potential = (torch.abs(distance_potential) + torch.abs(demand_potential)) / max_potential

    # Heuristic function that combines distance and demand potential
    heuristic_values = normalized_potential * (1 - demands_normalized)  # Priority to visit nodes with lower demand first

    return heuristic_values