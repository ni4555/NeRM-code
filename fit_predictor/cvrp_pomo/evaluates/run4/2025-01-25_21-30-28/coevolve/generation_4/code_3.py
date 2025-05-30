import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = torch.sum(demands)
    
    # Normalize demands
    normalized_demands = demands / total_demand
    
    # Calculate the difference in demand from the average demand
    demand_diff = normalized_demands - demands.mean()
    
    # Calculate the distance weighted by demand difference
    weighted_distance = distance_matrix * demand_diff
    
    # Use a sigmoid function to convert the weighted distance into a heuristic
    # This will ensure that edges with a high negative product (promising) have high positive heuristics
    heuristics = torch.sigmoid(weighted_distance)
    
    return heuristics