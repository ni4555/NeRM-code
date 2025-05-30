import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum demand to normalize the demands vector
    max_demand = torch.max(demands)
    # Normalize the demands to be between 0 and 1
    normalized_demands = demands / max_demand
    # Calculate the demand-based heuristic
    demand_heuristic = normalized_demands.unsqueeze(1) - normalized_demands.unsqueeze(0)
    
    # Calculate the distance-based heuristic
    # Subtract the minimum distance to the depot for each edge
    min_distance = torch.min(distance_matrix, dim=1, keepdim=True)[0]
    distance_heuristic = distance_matrix - min_distance
    
    # Combine the demand and distance heuristics
    combined_heuristic = demand_heuristic + distance_heuristic
    
    # Apply a small epsilon to avoid division by zero
    epsilon = 1e-6
    combined_heuristic = combined_heuristic + epsilon
    
    # Ensure that all negative values are set to zero to represent undesirable edges
    combined_heuristic = torch.clamp(combined_heuristic, min=0)
    
    return combined_heuristic