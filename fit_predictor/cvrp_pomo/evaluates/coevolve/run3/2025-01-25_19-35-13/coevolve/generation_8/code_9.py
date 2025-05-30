import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the negative demand heuristic
    negative_demand_heuristic = -normalized_demands
    
    # Calculate the distance heuristic
    distance_heuristic = distance_matrix
    
    # Combine the two heuristics by element-wise addition
    combined_heuristic = negative_demand_heuristic + distance_heuristic
    
    return combined_heuristic