import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum distance to the depot for each customer
    max_distance_to_depot = distance_matrix[:, 0].max(dim=0)[0]
    
    # Calculate the maximum demand that can be covered by the vehicle for each customer
    max_demand_coverable = demands / demands.sum()
    
    # Create a heuristic value based on the ratio of the maximum distance to the depot to the maximum demand coverable
    heuristic_values = max_distance_to_depot / max_demand_coverable
    
    # Normalize the heuristic values to ensure they are between -1 and 1
    min_val = heuristic_values.min()
    max_val = heuristic_values.max()
    normalized_values = (heuristic_values - min_val) / (max_val - min_val)
    
    # Adjust the heuristic values to be negative for undesirable edges and positive for promising ones
    adjusted_values = 2 * normalized_values - 1
    
    return adjusted_values