import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize node distances
    max_distance = distance_matrix.max()
    normalized_distance_matrix = distance_matrix / max_distance

    # Normalize demands
    max_demand = demands.max()
    normalized_demands = demands / max_demand

    # Calculate potential values for explicit depot handling
    depot_potential = torch.sum(normalized_distance_matrix, dim=1) - torch.sum(normalized_distance_matrix, dim=0)

    # Calculate a simple heuristic for each edge based on distance and demand
    heuristic_values = normalized_distance_matrix + normalized_demands

    # Adjust heuristic values based on depot potential
    heuristic_values = heuristic_values - depot_potential[:, None]

    # Introduce a penalty for high demand edges
    high_demand_penalty = torch.where(normalized_demands > 1, -10, 0)
    heuristic_values += high_demand_penalty

    return heuristic_values