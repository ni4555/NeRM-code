import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demands between each pair of nodes
    diff_demands = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Create a mask where the difference in demands is positive (i.e., a customer is needed)
    demand_mask = diff_demands > 0
    
    # Calculate the cost of visiting each customer
    # For this example, we will use the negative of the distance as a cost
    # A lower cost means a more promising edge
    cost_matrix = -distance_matrix
    
    # Combine the demand mask with the cost matrix
    # The result will be a matrix with negative values for promising edges
    heuristics_matrix = cost_matrix * demand_mask
    
    # Set the diagonal to zero, as visiting the same node twice is not possible
    torch.fill_diagonal_(heuristics_matrix, 0)
    
    return heuristics_matrix