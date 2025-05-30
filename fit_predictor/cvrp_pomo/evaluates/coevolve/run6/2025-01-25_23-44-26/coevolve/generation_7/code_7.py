import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix by the maximum distance to ensure non-zero values
    normalized_distance_matrix = distance_matrix / torch.max(distance_matrix)
    
    # Normalize the demands to be between 0 and 1
    normalized_demands = demands / torch.sum(demands)
    
    # Calculate the inverse distance heuristic
    inverse_distance_heuristic = 1 / normalized_distance_matrix
    
    # Integrate a demand-penalty mechanism
    demand_penalty = 1 - demands
    
    # Combine the heuristics using a weighted sum
    # The weights can be adjusted as needed
    weight_inverse_distance = 0.7
    weight_demand_penalty = 0.3
    combined_heuristic = weight_inverse_distance * inverse_distance_heuristic + weight_demand_penalty * demand_penalty
    
    return combined_heuristic