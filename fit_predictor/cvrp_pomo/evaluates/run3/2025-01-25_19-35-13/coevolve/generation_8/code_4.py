import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the negative demand heuristic
    heuristics_matrix += -normalized_demands.unsqueeze(1)
    heuristics_matrix += -normalized_demands.unsqueeze(0)
    
    # Calculate the distance heuristic
    heuristics_matrix += distance_matrix
    
    # Ensure that the diagonal elements (self-loops) are not included
    heuristics_matrix.fill_diagonal_(0)
    
    return heuristics_matrix