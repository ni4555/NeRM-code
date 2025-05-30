import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demand_mask = torch.zeros_like(distance_matrix)
    capacity_mask = torch.zeros_like(distance_matrix)

    # Demand Normalization
    demand_normalized = demands / total_capacity

    # Cumulative Demand Mask
    for i in range(n):
        for j in range(n):
            if i != j:
                cumulative_demand = demand_normalized[i].sum() + demand_normalized[j]
                demand_mask[i, j] = cumulative_demand

    # Edge Feasibility Mask
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_demand = demand_normalized[i].sum() + demand_normalized[j]
                if edge_demand <= 1.0:  # Assuming vehicle capacity is 1 for simplicity
                    capacity_mask[i, j] = 1.0

    # Clear Edge Evaluation
    edge_value = -distance_matrix * capacity_mask

    # Optimization Strategies
    # Here we simply return the negative of the distance matrix multiplied by the capacity mask
    # This prioritizes edges with lower distances and higher capacity feasibility
    return edge_value