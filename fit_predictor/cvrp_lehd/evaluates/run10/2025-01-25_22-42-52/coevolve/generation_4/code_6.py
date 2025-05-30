import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the demand ratio for each customer
    demand_ratio = demands / demands.sum()
    
    # Calculate the cost of serving each customer (negative for better, positive for worse)
    cost_matrix = -distance_matrix
    
    # Use the demand ratio to weight the cost matrix
    weighted_cost_matrix = cost_matrix * demand_ratio
    
    # Normalize by the maximum demand ratio to ensure all values are between 0 and 1
    normalized_weighted_cost_matrix = weighted_cost_matrix / (demand_ratio.max().item() + 1e-6)
    
    # Ensure that the heuristics matrix is not dominated by the largest value
    heuristics_matrix = normalized_weighted_cost_matrix - (normalized_weighted_cost_matrix.max().item() + 1e-6)
    
    return heuristics_matrix