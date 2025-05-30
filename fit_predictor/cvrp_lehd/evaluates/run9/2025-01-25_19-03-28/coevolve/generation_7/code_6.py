import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Compute the total vehicle capacity
    total_capacity = demands.sum()

    # Normalize demands by total capacity
    normalized_demands = demands / total_capacity

    # Calculate the potential cost for each edge
    # The heuristic function is a combination of:
    # - The negative of the distance (to prefer shorter routes)
    # - The product of the distance and the absolute demand (to prioritize heavier loads)
    # - The normalized demand (to encourage balanced vehicle loads)
    potential_costs = -distance_matrix + distance_matrix * torch.abs(normalized_demands)

    # Add a small constant to avoid division by zero when taking the log
    small_constant = 1e-6
    potential_costs += small_constant

    # Calculate the heuristic values by taking the log of the potential costs
    # and dividing by the maximum value to normalize
    max_potential_cost = potential_costs.max()
    heuristics = torch.log(potential_costs) / max_potential_cost

    return heuristics