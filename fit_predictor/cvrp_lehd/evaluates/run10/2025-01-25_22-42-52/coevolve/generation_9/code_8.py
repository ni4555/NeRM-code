import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to be between 0 and 1
    normalized_demands = demands / demands.sum()
    
    # Compute the sum of demands for each edge, which represents the total demand on the edge
    edge_demands = distance_matrix * normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Calculate the cost for each edge (for simplicity, using a simple cost function that is the negative of the demand)
    edge_costs = -edge_demands
    
    # The heuristic is a combination of the cost and a function that promotes short edges
    # For simplicity, we'll use the inverse of the distance as the promotion factor
    promotion_factor = 1.0 / (distance_matrix + 1e-6)  # Adding a small constant to avoid division by zero
    heuristics = edge_costs + promotion_factor
    
    return heuristics