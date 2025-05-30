import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate cumulative demand mask
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Normalize cumulative demand by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demand = cumulative_demand / total_capacity
    
    # Create edge feasibility mask
    edge_feasibility_mask = distance_matrix < normalized_demand
    
    # Initialize the heuristics matrix with negative values (undesirable edges)
    heuristics = -torch.ones_like(distance_matrix)
    
    # Replace negative values with positive ones for promising edges
    heuristics[edge_feasibility_mask] = torch.ones_like(edge_feasibility_mask)
    
    return heuristics