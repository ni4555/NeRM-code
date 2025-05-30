import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum
    
    # Calculate initial heuristic based on demand
    initial_heuristic = normalized_demands.unsqueeze(1) * distance_matrix.unsqueeze(0)
    
    # Adjust heuristic based on distance (more distant edges are less desirable)
    adjusted_heuristic = initial_heuristic - distance_matrix
    
    # Subtract minimum value in each column to shift the range of values
    adjusted_heuristic = adjusted_heuristic - adjusted_heuristic.min(dim=0, keepdim=True)[0]
    
    # Scale heuristic values to have a suitable range (e.g., [-1, 1] or [0, 1])
    # Here we scale to be between 0 and 1
    adjusted_heuristic = (adjusted_heuristic + adjusted_heuristic.abs().max()) / (2 * adjusted_heuristic.abs().max())
    
    return adjusted_heuristic