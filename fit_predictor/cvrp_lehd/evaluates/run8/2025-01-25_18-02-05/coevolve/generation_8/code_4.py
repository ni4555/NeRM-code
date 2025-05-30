import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming distance_matrix is a torch.Tensor of shape (n, n) and demands is a torch.Tensor of shape (n,)
    # where n is the number of nodes (including the depot node at index 0).
    
    # Normalize the demands by the sum of demands to get the demand per unit capacity
    demand_per_capacity = demands / demands.sum()
    
    # Compute the potential value of each edge (distance * demand per unit capacity)
    # This is a measure of how good it is to take an edge into account
    potential_value = distance_matrix * demand_per_capacity
    
    # The heuristic value is the potential value minus the demand at the destination node
    # Since we are considering the depot as node 0, we do not add the demand of the depot itself
    # and we subtract the demand of each customer node.
    heuristic_values = potential_value - demands
    
    # We want negative values for undesirable edges and positive for promising ones
    # To do this, we can take the absolute value of the heuristic values
    # This step makes sure that all values are non-negative, and we can interpret
    # them as a measure of how good it is to include an edge in the solution.
    heuristics = torch.abs(heuristic_values)
    
    return heuristics