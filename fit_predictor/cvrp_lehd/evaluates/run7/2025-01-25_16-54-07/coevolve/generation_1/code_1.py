import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Compute the heuristics based on the normalized demands and distance matrix
    # We use a simple heuristic: the attractiveness of an edge is inversely proportional to the distance
    # and directly proportional to the demand at the destination node
    heuristics = (1 / distance_matrix) * normalized_demands
    
    # Ensure the heuristics have the correct sign convention (negative for undesirable edges)
    # by adding a large positive constant to all values, and then subtracting the total demand.
    # This effectively inverts the scale so that higher heuristics correspond to more promising edges.
    large_positive_constant = 1e6
    heuristics = large_positive_constant * (1 / distance_matrix) - (large_positive_constant * total_demand * normalized_demands)
    
    return heuristics