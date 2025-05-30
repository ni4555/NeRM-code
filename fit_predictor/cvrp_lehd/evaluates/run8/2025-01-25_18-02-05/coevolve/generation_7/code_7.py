import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity (assuming it's 1 for simplicity)
    # This normalization is a placeholder, as the actual normalization would depend on the specific problem context.
    normalized_demands = demands / demands.sum()
    
    # Calculate the negative distance heuristic (shorter paths are more promising)
    negative_distance = -distance_matrix
    
    # Calculate the demand heuristic (encourage paths with loads closer to vehicle capacity)
    # Here we use a simple heuristic where higher demand per distance ratio is more promising
    demand_per_distance = demands / distance_matrix
    
    # Combine the heuristics with a simple linear combination
    # The coefficients (alpha and beta) can be tuned for different problem instances
    alpha, beta = 0.5, 0.5
    combined_heuristic = alpha * negative_distance + beta * demand_per_distance
    
    # Ensure that we don't include negative values in the heuristic matrix
    combined_heuristic = torch.clamp(combined_heuristic, min=0)
    
    return combined_heuristic