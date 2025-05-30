import random
import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demands
    normalized_demands = demands / demands.sum()
    
    # Calculate the negative of the demand for each edge to penalize high demand edges
    demand_penalty = -normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Calculate the inverse of the distance matrix as a heuristic (shorter distances are better)
    # Note: We add a small constant to avoid division by zero
    distance_heuristic = torch.inverse(distance_matrix + 1e-6)
    
    # Add a load balancing factor by considering the difference between demands
    load_balance = (demands.unsqueeze(1) - demands.unsqueeze(0)).abs()
    
    # Use a dynamic time factor that considers both time of day and expected traffic
    time_of_day = torch.rand(distance_matrix.shape[0]) * 24  # Simulate different hours of the day
    traffic_factor = torch.rand(distance_matrix.shape[0]) * 2 - 1  # Simulate expected traffic (0-1 range)
    dynamic_time_factor = (1 / (1 + time_of_day.unsqueeze(1) * time_of_day.unsqueeze(0) + traffic_factor))
    
    # Combine the demand penalty, distance heuristic, load balance, and dynamic time heuristic
    combined_heuristic = demand_penalty + distance_heuristic + load_balance * dynamic_time_factor
    
    # Introduce randomness with a controlled factor to escape local optima
    random_factor = torch.rand(distance_matrix.shape) * 0.05 - 0.025  # Slightly negative for diversity
    diversity_heuristic = combined_heuristic + random_factor
    
    # Non-linear transformation to amplify the impact of certain factors
    heuristics_non_linear = torch.relu(diversity_heuristic)
    
    return heuristics_non_linear
