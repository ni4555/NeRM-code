import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity as the sum of demands (excluding the depot demand)
    total_capacity = demands.sum().item()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Compute the difference between the normalized demands and 0.5
    demand_diff = normalized_demands - 0.5
    
    # Use the absolute value of the demand difference as a heuristic for each edge
    # Negative values for undesirable edges and positive values for promising ones
    edge_heuristics = torch.abs(demand_diff)
    
    return edge_heuristics