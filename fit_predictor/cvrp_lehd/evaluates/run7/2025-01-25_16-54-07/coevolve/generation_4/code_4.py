import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands vector does not include the depot demand
    demands = demands[1:]
    
    # Calculate the cumulative demand along each edge
    cumulative_demand = demands.cumsum(dim=0)
    
    # Initialize the heuristic matrix with high negative values (undesirable edges)
    heuristic_matrix = -torch.ones_like(distance_matrix)
    
    # Set the heuristic values for the depot to customer edges
    # The heuristic value is the negative of the cumulative demand at the customer
    heuristic_matrix[:, 1:] = -cumulative_demand
    
    # Set the heuristic values for the customer to depot edge
    # The heuristic value is the negative of the total demand of the customer
    heuristic_matrix[1:, 0] = -demands
    
    return heuristic_matrix