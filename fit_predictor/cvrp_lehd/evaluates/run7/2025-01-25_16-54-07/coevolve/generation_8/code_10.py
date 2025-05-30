import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    cumulative_demand_mask = torch.cumsum(demands, dim=0) / demands.sum()  # Normalize cumulative demand
    edge_feasibility_mask = (distance_matrix < demands[:, None]) * (distance_matrix < 1)  # Only consider edges with capacity left
    
    # Calculate edge contribution to balanced load distribution
    edge_contribution = (cumulative_demand_mask * edge_feasibility_mask).sum(dim=1) - cumulative_demand_mask
    
    # Create heuristic values, higher positive values indicate more promising edges
    heuristics = edge_contribution + (1 - edge_feasibility_mask.float()).sum(dim=1)  # Adding penalty for infeasible edges
    
    return heuristics