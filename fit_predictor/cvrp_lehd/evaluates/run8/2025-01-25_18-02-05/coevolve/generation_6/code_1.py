import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the heuristic value for each edge
    # A promising edge could be defined as having a negative heuristic
    # For simplicity, we use the negative of the distance as the heuristic
    # This assumes that shorter distances are preferable, which is common in routing problems
    
    # Note: The following assumes that the distance matrix and demands are on the same device
    # If not, you may need to move one to the device of the other before proceeding
    
    # Negative of the distance matrix for promising edges
    negative_distance_matrix = -distance_matrix
    
    # Add the demands to the negative distance matrix for the heuristic
    # This is a simple heuristic where we consider customer demand as an additional factor
    # We assume that higher demands (which are normalized) increase the "prominence" of the edge
    # The exact way to incorporate demand might depend on the problem specifics and the heuristic design
    demand_factor = demands.expand_as(distance_matrix)
    heuristic_matrix = negative_distance_matrix + demand_factor
    
    return heuristic_matrix