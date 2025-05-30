import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands to the range [0, 1]
    normalized_demands = demands / demands.sum()
    
    # Calculate the inverse distance heuristic
    inv_distance = 1.0 / distance_matrix
    
    # Calculate the load balancing heuristic
    load_balancing = torch.abs(normalized_demands - 0.5)
    
    # Combine the two heuristics
    combined_heuristics = inv_distance + load_balancing
    
    # Adjust the heuristics based on the total vehicle capacity
    # Here we assume the capacity is a scalar value
    vehicle_capacity = 1.0  # Replace with actual vehicle capacity if provided
    adjusted_heuristics = combined_heuristics * vehicle_capacity
    
    return adjusted_heuristics