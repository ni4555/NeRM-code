import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demand_threshold = total_capacity / n  # Normalize demand threshold by number of nodes
    
    # Calculate savings for each edge
    savings_matrix = torch.clamp(distance_matrix - 2 * demand_threshold, min=0)
    
    # Add demand-based savings (heuristic)
    savings_matrix += (demands[:, None] - demands[None, :])
    
    # Normalize the savings matrix to have a range of -1 to 1
    min_savings, max_savings = savings_matrix.min(), savings_matrix.max()
    savings_matrix = (savings_matrix - min_savings) / (max_savings - min_savings)
    
    # Invert the savings matrix to have negative values for undesirable edges
    heuristics = 1 - savings_matrix
    
    return heuristics