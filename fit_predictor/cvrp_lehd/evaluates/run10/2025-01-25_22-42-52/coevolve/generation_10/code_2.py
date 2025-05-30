import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize demands
    normalized_demands = demands / total_demand
    
    # Compute the cost of traveling from the depot to each customer
    cost_to_customers = distance_matrix[0, 1:]
    
    # Compute the cost of serving each customer considering their demand
    cost_with_demand = cost_to_customers * normalized_demands
    
    # Negative values for undesirable edges and positive values for promising ones
    heuristics = -cost_with_demand
    
    return heuristics