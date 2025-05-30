import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix to ensure that the shortest distance is 0
    distance_matrix = distance_matrix - distance_matrix.min()
    
    # Normalize the demands by the total vehicle capacity to ensure that
    # the demand is between 0 and 1
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the negative weighted distance based on the normalized demand
    # We use negative values to prioritize edges with lower demand
    negative_weighted_distance = -distance_matrix * normalized_demands
    
    # Apply dynamic load balancing by adjusting weights based on the proximity
    # to the depot. This encourages routes that start from the depot.
    depot_index = 0
    distance_to_depot = distance_matrix[depot_index]
    distance_to_depot[distance_to_depot == float('inf')] = 0  # Replace inf with 0 for calculation
    dynamic_load_balancing = negative_weighted_distance * (1 / (distance_to_depot + 1e-6))  # Add a small constant to avoid division by zero
    
    # Combine all the factors to get the final heuristic values
    heuristic_values = dynamic_load_balancing
    
    return heuristic_values