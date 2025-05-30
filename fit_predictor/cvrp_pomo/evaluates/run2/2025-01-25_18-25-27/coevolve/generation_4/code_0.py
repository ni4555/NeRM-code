import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Compute the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands to the total capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the demand-based heuristic for each edge
    demand_heuristic = distance_matrix * normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Calculate the distance-based heuristic for each edge
    distance_heuristic = distance_matrix
    
    # Combine the two heuristics using a weighted sum, with demand heuristic having higher weight
    combined_heuristic = demand_heuristic * 0.5 + distance_heuristic * 0.5
    
    # Adjust the heuristics to have negative values for undesirable edges and positive values for promising ones
    combined_heuristic = combined_heuristic - combined_heuristic.max()
    
    return combined_heuristic