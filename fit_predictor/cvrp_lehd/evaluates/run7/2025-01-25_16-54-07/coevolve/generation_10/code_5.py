import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Demand normalization: sum of demands divided by the total vehicle capacity
    vehicle_capacity = demands.sum()
    normalized_demands = demands / vehicle_capacity

    # Create cumulative demand mask for route prioritization
    cumulative_demand_mask = torch.cumsum(normalized_demands, dim=0)

    # Create edge feasibility mask for capacity constraint prioritization
    edge_feasibility_mask = torch.triu(torch.ones_like(distance_matrix), diagonal=1)
    edge_feasibility_mask[torch.nonzero(distance_matrix)[:, 1] >= edge_feasibility_mask[:, 0].sum(1)] = 0

    # Define a clear and objective evaluation method for edges
    edge_evaluation = -distance_matrix * cumulative_demand_mask * edge_feasibility_mask

    # Optimization strategies: simplify the evaluation matrix by considering the total distance and load
    # Add negative weight to edges that would exceed capacity
    total_demand_per_edge = torch.triu(torch.sum(demands[:, None] * edge_feasibility_mask, dim=1), diagonal=1)
    edge_evaluation[torch.nonzero(total_demand_per_edge > 1)] -= 1e4

    return edge_evaluation