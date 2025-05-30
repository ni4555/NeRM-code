import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Normalize the distance matrix by dividing by the maximum distance
    max_distance = distance_matrix.max()
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Create a matrix with the normalized demands subtracted from the normalized distances
    # This encourages edges to be selected if the demand at the destination is less than the distance
    demand_diff = normalized_distance_matrix - normalized_demands.unsqueeze(1)
    
    # Subtract the total capacity to ensure no route exceeds the vehicle's carrying capacity
    # This will make routes with high demand less promising
    demand_diff -= total_capacity
    
    # Use a threshold to set negative values to -1 and positive values to 1
    threshold = 0
    heuristics = torch.where(demand_diff < threshold, -1.0, 1.0)
    
    return heuristics