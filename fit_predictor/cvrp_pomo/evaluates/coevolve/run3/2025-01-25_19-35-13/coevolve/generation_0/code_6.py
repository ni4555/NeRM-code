import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Calculate the average demand per vehicle
    avg_demand = total_demand / len(demands)
    
    # Calculate the maximum deviation from average demand for each node
    max_deviation = torch.abs(demands - avg_demand)
    
    # Create a matrix with 0s for the diagonal (no self-loops) and 1s otherwise
    edge_mask = (distance_matrix != 0).float()
    
    # Calculate the heuristic values based on the maximum deviation and edge mask
    heuristics = max_deviation * edge_mask
    
    return heuristics