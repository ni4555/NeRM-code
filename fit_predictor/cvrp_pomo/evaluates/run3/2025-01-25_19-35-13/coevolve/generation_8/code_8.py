import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Demand normalization
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the potential cost for each edge
    # Assuming distance_matrix is symmetric for simplicity
    potential_costs = distance_matrix * normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)

    # Multi-objective fitness function
    # We want to minimize the total distance, so we use negative potential costs
    heuristics = -potential_costs.sum(dim=(1, 2))

    return heuristics