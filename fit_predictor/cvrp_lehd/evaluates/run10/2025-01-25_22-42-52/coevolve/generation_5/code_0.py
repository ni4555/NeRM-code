import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Compute the heuristic values based on the normalized demands
    # Negative values for undesirable edges (e.g., if demand is 0 or too high)
    # Positive values for promising edges (e.g., if demand is within a reasonable range)
    # We will use a simple heuristic where we consider edges with demand within 0.5 to 1.5 times the average demand as promising
    average_demand = normalized_demands.mean()
    heuristics = -torch.where(normalized_demands < 0.5, torch.ones_like(normalized_demands), torch.zeros_like(normalized_demands))
    heuristics = torch.where(normalized_demands > 1.5, -torch.ones_like(normalized_demands), heuristics)
    heuristics = torch.where((normalized_demands >= 0.5) & (normalized_demands <= 1.5), 1.0, heuristics)
    
    # Adjust the heuristics based on the distance matrix
    # The idea is to add a small positive value to shorter distances, which could be seen as more promising
    heuristics += distance_matrix / distance_matrix.max()
    
    return heuristics