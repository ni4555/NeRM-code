import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    """Improved heuristic function that considers both distance and demand, with normalization for diversity."""
    # Normalize demands to a range of 0 to 1 based on the maximum demand
    max_demand = demands.max()
    normalized_demands = demands / max_demand
    
    # Normalize distances to a range of 0 to 1 based on the maximum distance
    max_distance = distance_matrix.max()
    normalized_distances = distance_matrix / max_distance
    
    # Calculate the negative demand penalty
    demand_penalty = -normalized_demands
    
    # Combine the distance and demand penalties
    combined_penalties = normalized_distances + demand_penalty
    
    # Normalize the combined penalties to a range of -1 to 1
    min_val = combined_penalties.min()
    max_val = combined_penalties.max()
    heuristics = 2 * (combined_penalties - min_val) / (max_val - min_val) - 1
    
    return heuristics
