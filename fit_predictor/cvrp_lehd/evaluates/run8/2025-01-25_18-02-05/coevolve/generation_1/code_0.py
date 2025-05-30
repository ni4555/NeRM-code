import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize them
    total_demand = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_demand
    
    # Calculate the potential value for each edge based on normalized demand
    demand_potential = normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Calculate the cost for each edge based on distance
    cost = distance_matrix
    
    # Combine demand potential and cost to get the heuristic value
    heuristic_values = demand_potential - cost
    
    # Set negative values to a very small number to indicate undesirable edges
    undesirable_threshold = -1e-5
    heuristic_values[heuristic_values < undesirable_threshold] = undesirable_threshold
    
    return heuristic_values