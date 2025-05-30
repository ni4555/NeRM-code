import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the heuristic values based on the ratio of demand to distance
    # This heuristic assumes that closer nodes with higher demand are more promising
    heuristics = demands / distance_matrix
    
    # Normalize the heuristic values to ensure they are between -1 and 1
    max_demand = demands.max()
    min_demand = demands.min()
    normalized_demand = (demands - min_demand) / (max_demand - min_demand)
    
    # Calculate the heuristic values based on the normalized demand
    normalized_heuristics = normalized_demand / distance_matrix
    
    # Convert to a range between -1 and 1
    max_distance = distance_matrix.max()
    min_distance = distance_matrix.min()
    normalized_distance = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    # Create the final heuristics matrix
    heuristics_matrix = normalized_demand * normalized_distance
    
    # Apply a simple penalty for edges leading back to the depot (which should be avoided)
    penalty = torch.zeros_like(distance_matrix)
    penalty[torch.arange(distance_matrix.shape[0]), torch.arange(distance_matrix.shape[0])] = -1
    heuristics_matrix += penalty
    
    return heuristics_matrix