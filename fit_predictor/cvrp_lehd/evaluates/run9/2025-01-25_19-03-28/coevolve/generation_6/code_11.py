import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()

    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity

    # Calculate the potential cost of each edge
    potential_costs = distance_matrix * normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)

    # Calculate the heuristic values for each edge
    # We use a simple heuristic that promotes edges with smaller potential costs
    heuristics = potential_costs.sum(dim=2) - potential_costs.sum(dim=1)

    return heuristics