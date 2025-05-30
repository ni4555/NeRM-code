import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand at each node including the depot
    cumulative_demand = demands.cumsum(dim=0)
    
    # Calculate the potential heuristics based on the difference in cumulative demand
    # We use a simple heuristic that considers the difference in demand between nodes
    heuristics = (cumulative_demand[1:] - cumulative_demand[:-1]) * (distance_matrix[1:] - distance_matrix[:-1])
    
    # Normalize the heuristics by the maximum absolute value to ensure a balanced scale
    max_abs_value = torch.max(torch.abs(heuristics))
    heuristics = heuristics / max_abs_value
    
    # Apply a threshold to make the heuristics binary: positive for promising edges, negative for undesirable ones
    threshold = 0.5
    heuristics = torch.where(heuristics > threshold, 1.0, -1.0)
    
    return heuristics