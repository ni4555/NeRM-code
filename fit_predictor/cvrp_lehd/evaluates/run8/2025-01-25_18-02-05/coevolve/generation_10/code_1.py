import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity (assuming demands are already normalized)
    
    # The heuristic will be a linear combination of the negative distance and demands
    # For simplicity, let's assume we weigh distance more than demand (weight can be adjusted)
    weight_distance = 0.8
    weight_demand = 0.2
    
    # Calculate the negative distance matrix to give preference to shorter distances
    negative_distance = -distance_matrix
    
    # Calculate the negative demand to give preference to customers with lower demand
    negative_demand = -demands
    
    # Compute the heuristic values as a weighted sum of the negative distance and demand
    heuristic_values = weight_distance * negative_distance + weight_demand * negative_demand
    
    # Ensure that undesirable edges have negative values and promising ones have positive values
    # We do this by adding the maximum value of the demand vector to the negative distance
    # This ensures that all edges have positive values, with larger values indicating better routes
    max_demand = torch.max(negative_demand)
    heuristic_values = heuristic_values + max_demand
    
    return heuristic_values