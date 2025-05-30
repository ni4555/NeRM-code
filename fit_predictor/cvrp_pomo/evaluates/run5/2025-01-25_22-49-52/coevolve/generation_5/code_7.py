import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = torch.sum(demands)
    demands_normalized = demands / total_capacity

    # Calculate potential based on distance and demand
    distance_potential = distance_matrix.clone()
    demand_potential = demands_normalized.repeat(n, 1)
    path_potential = distance_potential * demand_potential

    # Normalize path potential for consistent scaling
    path_potential = torch.nn.functional.normalize(path_potential, p=1, dim=1)

    # Adjust potential to promote load balancing
    load_balance_factor = torch.clamp((1 - demands_normalized), min=0, max=1)
    balanced_potential = path_potential * load_balance_factor

    # Heuristic function that promotes edges with higher balanced potential
    heuristic_matrix = balanced_potential.clone()
    # Apply a negative heuristic value for undesirable edges (e.g., high distance)
    # This is a simple heuristic to avoid long distances, it can be refined
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristic_matrix[i, j] = -torch.clamp(distance_matrix[i, j], min=0, max=1)
            else:
                heuristic_matrix[i, j] = 0

    return heuristic_matrix