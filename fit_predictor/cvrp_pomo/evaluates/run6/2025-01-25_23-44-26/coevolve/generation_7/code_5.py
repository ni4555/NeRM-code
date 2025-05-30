import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix
    max_distance = torch.max(distance_matrix)
    normalized_distance_matrix = distance_matrix / max_distance

    # Calculate the inverse distance heuristic
    inverse_distance_matrix = 1 / normalized_distance_matrix

    # Normalize the demands
    max_demand = torch.max(demands)
    normalized_demands = demands / max_demand

    # Calculate the demand-penalty matrix
    demand_penalty_matrix = normalized_demands - 1

    # Combine the inverse distance and demand-penalty heuristics
    combined_heuristic_matrix = inverse_distance_matrix - demand_penalty_matrix

    # Apply a slight positive offset to ensure non-zero weights
    combined_heuristic_matrix = combined_heuristic_matrix + 0.001

    return combined_heuristic_matrix