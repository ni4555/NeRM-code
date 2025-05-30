import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total distance from the depot to all other nodes
    total_distance_from_depot = distance_matrix[0, 1:]
    
    # Calculate the total distance from all nodes back to the depot
    total_distance_to_depot = distance_matrix[:, 1:].sum(dim=1)
    
    # Calculate the sum of demands for each route
    sum_of_demands = demands[1:]
    
    # Calculate the heuristic value for each edge
    heuristic_values = (total_distance_from_depot - total_distance_to_depot) - sum_of_demands
    
    # The heuristic values are negative for undesirable edges, so we take the negative of the result
    return -heuristic_values