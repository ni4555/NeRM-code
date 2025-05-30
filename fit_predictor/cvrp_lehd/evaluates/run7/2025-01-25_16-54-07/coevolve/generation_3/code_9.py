import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the savings for each edge
    savings = distance_matrix.clone() - (distance_matrix.min(dim=1, keepdim=True)[0] + distance_matrix.min(dim=0, keepdim=True)[0])
    savings = savings * demands  # Multiply savings by demand to penalize longer routes
    
    # Normalize the savings by the maximum possible savings to create a promising indicator
    max_savings = savings.max()
    heuristics = savings / max_savings
    
    # Cap the heuristics values to ensure negative values for undesirable edges
    heuristics = torch.clamp(heuristics, min=-1.0, max=1.0)
    
    return heuristics