import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Create a tensor that will store the heuristic values for each edge
    n = distance_matrix.shape[0]
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic for each edge
    # We use a simple heuristic based on the demand of the destination node
    # Edges with higher demand are given lower values, hence they are considered less promising
    heuristic_matrix[:, 1:] = (1 - demands[1:]) / demands[1:]
    # The diagonal elements (self-loops) should be zero or negative (we can choose a negative value, like -1)
    # Here we set them to -1 to indicate they are not promising
    torch.fill_diagonal(heuristic_matrix, value=-1)
    
    return heuristic_matrix