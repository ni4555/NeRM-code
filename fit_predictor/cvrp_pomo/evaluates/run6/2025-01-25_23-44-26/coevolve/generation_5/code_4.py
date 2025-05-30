import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity (assuming total_capacity is given as a parameter)
    # For this example, we'll assume total_capacity is the sum of all demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Demand penalty function: higher demand closer to capacity limit gets a higher penalty
    demand_penalty = (1 - normalized_demands) * 1000

    # Inverse distance heuristic: closer nodes are more promising
    inverse_distance = 1 / distance_matrix

    # Combine demand penalty and inverse distance heuristic
    heuristic_values = inverse_distance - demand_penalty

    # Replace negative values with zeros (to avoid including undesirable edges)
    heuristic_values[heuristic_values < 0] = 0

    return heuristic_values