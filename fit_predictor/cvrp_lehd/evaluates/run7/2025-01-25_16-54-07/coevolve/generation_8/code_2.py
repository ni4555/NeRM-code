import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    
    # Calculate cumulative demand mask
    cumulative_demand = (demands.cumsum(dim=0) / total_capacity).unsqueeze(1)
    cumulative_demand = cumulative_demand.repeat(1, n)
    
    # Calculate edge feasibility mask
    edge_capacity_mask = (distance_matrix < 1e6) & (demands.unsqueeze(1) < 1e6)
    
    # Calculate load difference
    load_difference = (cumulative_demand * distance_matrix).sum(dim=1) - cumulative_demand.sum(dim=1)
    
    # Prioritize edges based on their contribution to balanced load distribution
    heuristics = load_difference * edge_capacity_mask.float()
    
    return heuristics