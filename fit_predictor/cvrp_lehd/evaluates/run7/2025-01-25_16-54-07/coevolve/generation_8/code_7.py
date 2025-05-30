import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Calculate the cumulative load distribution
    load_distribution = cumulative_demand / total_capacity
    
    # Create a cumulative demand mask
    cumulative_demand_mask = (load_distribution < 1).float()
    
    # Create an edge feasibility mask
    edge_feasibility_mask = (distance_matrix < total_capacity).float()
    
    # Calculate the priority of each edge based on its contribution to balanced load distribution
    priority = (cumulative_demand_mask * edge_feasibility_mask * (1 - load_distribution)).neg()
    
    return priority