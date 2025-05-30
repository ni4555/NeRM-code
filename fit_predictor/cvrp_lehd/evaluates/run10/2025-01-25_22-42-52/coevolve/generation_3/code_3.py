import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to the range [0, 1]
    demands_normalized = demands / demands.sum()
    
    # Calculate the potential cost for each edge
    potential_costs = distance_matrix * demands_normalized.unsqueeze(1) * demands.unsqueeze(0)
    
    # Adjust potential costs to make them more negative for undesirable edges
    adjusted_costs = potential_costs - (potential_costs.max() * 0.5)
    
    # Use a simple heuristic: the closer to zero, the more promising the edge
    heuristics = adjusted_costs
    
    return heuristics