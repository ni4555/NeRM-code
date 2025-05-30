import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Demand normalization
    total_demand = demands.sum()
    normalized_demands = demands / total_demand

    # Cumulative demand mask
    cumulative_demand = torch.cumsum(normalized_demands, dim=0)
    cumulative_demand[1:] -= cumulative_demand[:-1]
    cumulative_demand[0] = 0

    # Edge feasibility mask
    edge_capacity_mask = distance_matrix < demands.unsqueeze(1)
    edge_capacity_mask = edge_capacity_mask & edge_capacity_mask.transpose(0, 1)

    # Capacity-based prioritization
    capacity = torch.sum(distance_matrix)
    capacity_mask = distance_matrix < capacity.unsqueeze(1)
    capacity_mask = capacity_mask & capacity_mask.transpose(0, 1)

    # Clear edge evaluation
    edge_evaluation = -distance_matrix
    edge_evaluation = edge_evaluation * (capacity_mask & edge_capacity_mask)

    # Optimization strategies
    # Direct optimization technique: we use a simple heuristic where edges with lower distance and
    # higher demand are more promising. We adjust this by the cumulative demand and capacity constraints.
    edge_evaluation += cumulative_demand.unsqueeze(1) * (distance_matrix > 0)
    
    return edge_evaluation