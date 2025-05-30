import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Get the number of nodes
    n = distance_matrix.shape[0]
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the negative of the distance matrix for the heuristic values
    # Negative values are preferred in the heuristic as they represent "promise"
    heuristic_matrix = -distance_matrix
    
    # Incorporate demand considerations:
    # Edges to nodes with higher demands should be less promising
    demand_weight = (demands / demands.sum()).unsqueeze(1)
    heuristic_matrix = heuristic_matrix + torch.mul(demand_weight, distance_matrix)
    
    # Incorporate some form of service time considerations
    # For simplicity, we use a uniform factor, but this could be replaced with a more complex function
    service_time_factor = torch.ones_like(demand_weight)
    heuristic_matrix = heuristic_matrix + torch.mul(service_time_factor, distance_matrix)
    
    # Ensure the heuristic matrix has positive values for promising edges
    # and negative values for undesirable edges
    heuristic_matrix = torch.clamp(heuristic_matrix, min=-1e10, max=0)
    
    return heuristic_matrix