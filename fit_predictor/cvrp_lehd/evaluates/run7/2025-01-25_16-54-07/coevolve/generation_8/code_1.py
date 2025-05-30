import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    cumulative_demand_mask = torch.cumsum(demands[1:], dim=0) / demands.sum()
    capacity_remaining = demands * 2  # Assuming maximum load is double the vehicle capacity
    
    # Calculate edge feasibility mask
    edge_feasibility_mask = (distance_matrix < capacity_remaining[1:])
    
    # Calculate the contribution to balanced load distribution
    load_balance_contribution = (cumulative_demand_mask - demands[1:]) / n
    
    # Combine the feasibility and load balance contributions
    heuristics = edge_feasibility_mask * load_balance_contribution
    
    return heuristics