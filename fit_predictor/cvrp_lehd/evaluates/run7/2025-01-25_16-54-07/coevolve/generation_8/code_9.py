import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Calculate the cumulative demand mask
    cumulative_demand_mask = (cumulative_demand <= vehicle_capacity).float()
    
    # Calculate the edge feasibility mask
    edge_feasibility_mask = (distance_matrix != 0) * cumulative_demand_mask[:, None]
    
    # Calculate the contribution to balanced load distribution
    load_distribution_contribution = (vehicle_capacity - cumulative_demand) * edge_feasibility_mask
    
    # Prioritize edges based on their contribution
    edge_priority = load_distribution_contribution / load_distribution_contribution.sum(dim=1, keepdim=True)
    
    # Convert to negative values for undesirable edges and positive values for promising ones
    heuristics = (edge_priority - 1) * edge_feasibility_mask
    
    return heuristics