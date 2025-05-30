import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix by the maximum distance to prevent overflow
    max_distance = torch.max(distance_matrix)
    normalized_distance_matrix = distance_matrix / max_distance

    # Calculate the cumulative demand mask
    cumulative_demand_mask = torch.cumsum(demands, dim=0)

    # Calculate the load distribution heuristic based on normalized distance and cumulative demand
    load_distribution_heuristic = -normalized_distance_matrix + cumulative_demand_mask

    # Normalize the load distribution heuristic to ensure all values are within a certain range
    min_val = torch.min(load_distribution_heuristic)
    max_val = torch.max(load_distribution_heuristic)
    load_distribution_heuristic = (load_distribution_heuristic - min_val) / (max_val - min_val)

    return load_distribution_heuristic