import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative of the distance matrix to use for heuristic values
    # This is based on the assumption that shorter distances are more promising
    heuristic_matrix = -distance_matrix
    
    # Normalize the heuristic matrix by the demands to incorporate demand information
    # The demand-based heuristic encourages selecting edges that serve lower demand customers
    # The demands tensor is broadcasted across the matrix to compute the heuristic value for each edge
    heuristic_matrix = heuristic_matrix / demands[:, None]
    
    return heuristic_matrix