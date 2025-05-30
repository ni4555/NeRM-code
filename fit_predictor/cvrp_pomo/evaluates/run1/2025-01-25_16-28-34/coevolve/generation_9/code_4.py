import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    # Calculate the normalized demand difference between nodes
    demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Calculate the sum of absolute demand differences
    total_demand_diff = torch.abs(demand_diff).sum(dim=2)
    
    # Calculate the total distance matrix
    total_distance = torch.sum(distance_matrix, dim=2)
    
    # Create a heuristics matrix that balances distance and demand differences
    heuristics = (total_distance - total_demand_diff) * 0.5
    
    # Normalize the heuristics matrix by the maximum value
    heuristics = heuristics / torch.max(heuristics)
    
    return heuristics