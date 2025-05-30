import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of the demand vector to normalize
    demand_sum = demands.sum().item()
    
    # Normalize the demand vector
    normalized_demands = demands / demand_sum
    
    # Create a vector with a constant value to add to the product of demands and distance
    constant = 1.0 / demand_sum
    
    # Compute the product of normalized demands and distances
    demand_distance_product = (constant * demands.unsqueeze(1) * distance_matrix.unsqueeze(0)).clamp(min=0.1)
    
    # Compute the heuristics as the sum of the demand_distance_product and the normalized demands
    heuristics = demand_distance_product + normalized_demands
    
    return heuristics