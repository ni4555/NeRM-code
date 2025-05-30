import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Normalize the distance matrix by the maximum distance in the matrix
    max_distance = distance_matrix.max()
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Calculate the heuristic value for each edge
    # The heuristic value is the product of the normalized demand and the normalized distance
    heuristic_matrix = normalized_demands.unsqueeze(1) * normalized_distance_matrix.unsqueeze(0)
    
    # Add a penalty for edges that exceed the vehicle's carrying capacity
    # Here, we use a simple approach where we subtract the total capacity from the heuristic value
    # if the demand at the destination node exceeds the vehicle's carrying capacity
    for i in range(1, len(demands)):
        if demands[i] > total_capacity:
            heuristic_matrix[i] -= total_capacity
    
    return heuristic_matrix