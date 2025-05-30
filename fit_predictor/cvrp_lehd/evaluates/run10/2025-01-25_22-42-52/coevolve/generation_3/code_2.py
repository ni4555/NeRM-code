import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands
    total_capacity = demands.sum()
    
    # Normalize the demands to be between 0 and 1
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristics based on the normalized demands
    # For simplicity, we use a linear function of the normalized demands
    # The coefficient is chosen such that the heuristic value is positive for promising edges
    # and negative for undesirable edges
    heuristic_values = normalized_demands * 10
    
    # Adjust the heuristic values for the depot node
    # We set the heuristic value to be negative for the depot node edges to discourage them
    # from being included in the solution
    depot_index = 0
    heuristic_values[depot_index, :] = -1
    heuristic_values[:, depot_index] = -1
    
    return heuristic_values