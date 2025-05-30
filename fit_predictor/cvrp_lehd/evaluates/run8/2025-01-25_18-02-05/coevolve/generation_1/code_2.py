import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands for normalization
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics for each edge
    # We use the normalized demand as the heuristic value for each edge
    # since higher demand implies a more promising edge to include in the solution
    heuristics = distance_matrix * normalized_demands
    
    # To make the edge selection more heuristic, we can also incorporate the distance
    # by subtracting it from the heuristic value, which makes short distances more attractive
    heuristics -= distance_matrix
    
    # Ensure that the heuristics have negative values for undesirable edges
    # and positive values for promising ones
    heuristics[distance_matrix == 0] = 0  # Set the depot edges to zero
    heuristics[distance_matrix == float('inf')] = -float('inf')  # Set unreachable edges to a very negative value
    
    return heuristics