import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Node partitioning
    partition_indices = torch.argsort(torch.abs(demands))
    partition_sizes = torch.tensor([0] * n)
    for i in range(n):
        partition_sizes[partition_indices[i]] += 1
    
    # Demand relaxation
    relaxed_demands = demands.clone()
    for i in range(n):
        relaxed_demands[i] = min(1.0, relaxed_demands[i] * (partition_sizes[i] / demands.sum()))
    
    # Dynamic window approach
    dynamic_window = torch.zeros_like(relaxed_demands)
    for i in range(n):
        dynamic_window[i] = relaxed_demands[i] * 2
    
    # Multi-objective evolutionary algorithm (simplified)
    # Here we just use a simple heuristic: the more relaxed demand, the more promising the edge
    edge_promise = (dynamic_window - demands) / (dynamic_window + demands)
    
    # Apply constraint programming to enforce vehicle capacities
    # For simplicity, we use a basic heuristic that penalizes edges leading to overcapacity
    max_capacity = demands.sum()
    over_capacity_penalty = torch.clamp((demands * distance_matrix).sum(axis=1) - max_capacity, 0, float('inf'))
    edge_promise -= over_capacity_penalty
    
    return edge_promise