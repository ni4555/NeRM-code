import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands to get the fraction of vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Normalize the distance matrix to have a mean of 0 and std of 1
    mean_distance = distance_matrix.mean()
    std_distance = distance_matrix.std()
    normalized_distance = (distance_matrix - mean_distance) / std_distance
    
    # Create a matrix where the value for each edge is the sum of the normalized demand and distance
    # The heuristic value for each edge is negative for longer distances and higher demands (undesirable)
    # and positive for shorter distances and lower demands (promising)
    heuristic_matrix = -normalized_distance * normalized_demands
    
    # Check vehicle capacities: the sum of demands along any route should not exceed the vehicle capacity
    # We will ensure this by adding a penalty to the heuristic for edges that would result in exceeding capacity
    # We assume the first edge is the edge to the depot (0) and that it's the first in the route
    for i in range(1, len(demands)):
        # For each edge from the i-th customer to the (i+1)-th customer, add a penalty if the sum of demands would exceed capacity
        sum_demand = (demands[:i+1]).sum()
        if sum_demand > 1:
            heuristic_matrix[i][i+1] += sum_demand - 1
    
    return heuristic_matrix