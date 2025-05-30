import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand to normalize
    total_demand = demands.sum()
    # Normalize demands to get the share of capacity each customer demands
    normalized_demands = demands / total_demand
    # Calculate the potential value of each edge (distance times demand share)
    potential_values = distance_matrix * normalized_demands
    # We can add some penalties for edges with zero demand to avoid trivial solutions
    zero_demand_penalty = -1e4
    potential_values[torch.where(demands == 0)] = zero_demand_penalty
    return potential_values