import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demands by dividing each demand by the total capacity
    normalized_demands = demands / demands.sum()
    
    # Calculate the normalized distances by dividing each distance by the maximum distance
    max_distance = torch.max(distance_matrix)
    normalized_distances = distance_matrix / max_distance
    
    # Combine normalized demands and distances using a simple weighted sum
    # Weights are set to 0.5 for each, but these can be adjusted for different heuristic approaches
    combined_heuristics = 0.5 * normalized_demands.unsqueeze(1) + 0.5 * normalized_distances
    
    # Add a penalty for high demands, to encourage selecting edges with lower demands
    penalty_factor = 0.1
    demand_penalty = -penalty_factor * (demands - demands.mean()).unsqueeze(1)
    heuristics = combined_heuristics + demand_penalty
    
    return heuristics