import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the demand-based heuristic (promising edges will have higher values)
    demand_heuristic = normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Calculate the distance-based heuristic (undesirable edges will have lower values)
    # Here we use a simple inverse distance heuristic as an example
    distance_heuristic = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Combine the demand and distance heuristics with a penalty for high demand
    # and high distance, which are typically undesirable
    combined_heuristic = demand_heuristic - distance_heuristic
    
    # Ensure the combined heuristic has negative values for undesirable edges
    # and positive values for promising ones by adding a constant
    constant = total_capacity / 100  # This constant can be adjusted
    combined_heuristic = combined_heuristic + constant
    
    return combined_heuristic