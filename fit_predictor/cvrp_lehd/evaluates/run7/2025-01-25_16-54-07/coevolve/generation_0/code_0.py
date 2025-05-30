import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative distance from each customer to the depot (excluding the depot itself)
    negative_distances = -distance_matrix[1:]
    
    # Calculate the sum of demands excluding the depot to normalize by total capacity
    total_demand = demands[1:].sum()
    
    # Normalize negative distances by the total demand to give a measure of "prominence"
    normalized_negative_distances = negative_distances / total_demand
    
    # Calculate the sum of demands to normalize each demand
    normalized_demands = demands[1:] / total_demand
    
    # Calculate the heuristic value as a combination of normalized distance and demand
    # Promising edges have positive values (negative distance + small demand)
    heuristics = normalized_negative_distances + normalized_demands
    
    # Replace all zeros (which could be caused by division by zero) with a negative value
    # to indicate that the edge should not be considered (e.g., a demand of zero)
    heuristics[heuristics == 0] = -1
    
    # Set the depot node (0th index) to 0 because we don't want to start or end at the depot
    heuristics[0] = 0
    
    return heuristics