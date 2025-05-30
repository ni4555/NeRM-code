import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the savings for each edge (i, j)
    savings = 2 * demands * distance_matrix - demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Set the savings to be negative for the diagonal (no savings when visiting the same node)
    savings.diag().fill_(0)
    
    # Set the savings to be negative for edges where the savings are less than 0
    heuristics = savings - torch.clamp(savings, min=0)
    
    return heuristics