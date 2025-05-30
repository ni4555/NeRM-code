import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demand vector is a column vector
    demands = demands.view(-1, 1)
    
    # Calculate the total demand
    total_demand = demands.sum(0)
    
    # Compute the demand-based heuristic (using a simple inverse heuristic)
    # This heuristic assigns higher weights to edges with lower demand
    # to encourage the algorithm to fill the vehicles
    demand_based_heuristic = 1.0 / (demands + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Normalize the heuristic to the range of [0, 1]
    # This helps in balancing the influence of the heuristic with the distance
    max_demand_based_heuristic = demand_based_heuristic.max()
    normalized_heuristic = demand_based_heuristic / max_demand_based_heuristic
    
    # Optionally, you can combine this heuristic with the distance-based heuristic
    # to give a weighted sum of the two:
    # combined_heuristic = 0.5 * normalized_heuristic + 0.5 * distance_matrix
    
    return normalized_heuristic