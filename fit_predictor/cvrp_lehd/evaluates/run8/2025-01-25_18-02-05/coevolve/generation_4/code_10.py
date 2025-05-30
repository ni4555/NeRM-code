import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize demands to get the fraction of the vehicle capacity used by each customer
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic value for each edge based on the normalized demand of the destination
    # We use the negative of the normalized demand to penalize edges leading to customers with high demand
    # This is a simple heuristic where edges to more demanding customers are less desirable
    heuristics = -normalized_demands[distance_matrix.nonzero().transpose(0, 1)[1]]

    return heuristics