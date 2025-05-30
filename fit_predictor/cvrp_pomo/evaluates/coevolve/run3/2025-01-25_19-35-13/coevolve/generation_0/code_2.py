import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix
    distance_matrix = distance_matrix / torch.max(distance_matrix)
    
    # Calculate the sum of the demands
    total_demand = torch.sum(demands)
    
    # Normalize the demands by the total demand
    demands_normalized = demands / total_demand
    
    # Compute the potential negative value for undesirable edges
    undesirable_edges = -distance_matrix
    
    # Compute the potential positive value for promising edges
    promising_edges = (1 - demands_normalized) * distance_matrix
    
    # Combine the negative and positive values
    heuristics_matrix = undesirable_edges + promising_edges
    
    return heuristics_matrix