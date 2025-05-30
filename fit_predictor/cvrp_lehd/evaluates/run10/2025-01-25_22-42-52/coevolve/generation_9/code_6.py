import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to sum to 1
    normalized_demands = demands / demands.sum()
    
    # Initialize heuristics with a high value (indicating not desirable)
    heuristics = torch.full_like(distance_matrix, fill_value=1e10)
    
    # Calculate the "promise" of including each edge
    # This is a simple heuristic: a higher demand implies a higher "promise"
    for i in range(len(demands)):
        heuristics[i, :] *= 1 / demands[i]  # More demand, lower weight
        heuristics[:, i] *= 1 / demands[i]  # More demand, lower weight
    
    # Subtract the distance_matrix to have negative values for undesirable edges
    heuristics -= distance_matrix
    
    # Clamp negative values to 0, ensuring the return values are positive
    heuristics = torch.clamp(heuristics, min=0)
    
    return heuristics