import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand (demand divided by total capacity)
    normalized_demand = demands / demands.sum()
    
    # Calculate the total distance for each edge
    total_distance = distance_matrix.sum(dim=1)
    
    # Compute the heuristic value for each edge
    # The heuristic is a combination of the normalized demand and the total distance
    # We can use a simple linear combination here, e.g., 0.5 * demand + 0.5 * distance
    # You can adjust the coefficients as needed to fine-tune the heuristic
    heuristic_values = 0.5 * normalized_demand.unsqueeze(1) + 0.5 * total_distance.unsqueeze(0)
    
    # Return the computed heuristic values
    return heuristic_values