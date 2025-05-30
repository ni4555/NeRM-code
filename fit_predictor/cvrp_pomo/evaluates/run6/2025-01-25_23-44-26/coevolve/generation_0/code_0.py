import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands for each node
    demand_sum = demands.sum(dim=0)
    
    # Initialize a mask for edges that are undesirable
    undesirable_edges = (demand_sum > 1)  # Assuming 1 is the normalized vehicle capacity
    
    # Create a negative value for undesirable edges
    undesirable_value = torch.full_like(distance_matrix, fill_value=-1)
    
    # Create a positive value for promising edges
    promising_value = torch.zeros_like(distance_matrix)
    
    # Assign the negative value for undesirable edges
    distance_matrix[undesirable_edges] = undesirable_value[undesirable_edges]
    
    # The rest of the edges are promising, so assign a positive value
    promising_edges = ~undesirable_edges
    distance_matrix[promising_edges] = promising_value[promising_edges]
    
    return distance_matrix