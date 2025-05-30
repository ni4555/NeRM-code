import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demand vector by the total vehicle capacity
    vehicle_capacity = 1.0  # Assuming a unit capacity for the purpose of this example
    normalized_demands = demands / vehicle_capacity
    
    # Demand penalty function: higher demand customers have a higher penalty
    demand_penalty = normalized_demands * 1000  # Example penalty factor
    
    # Inverse distance heuristic: shorter distances have lower heuristics values
    inverse_distance = 1.0 / (distance_matrix + 1e-8)  # Adding a small value to avoid division by zero
    
    # Combine the inverse distance heuristic and demand penalty
    combined_heuristic = inverse_distance - demand_penalty
    
    # Ensure all values are within the range of -100 to 100 for edge selection
    combined_heuristic = torch.clamp(combined_heuristic, min=-100, max=100)
    
    return combined_heuristic