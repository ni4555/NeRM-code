import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix to account for the fact that we want to consider
    # the distance between nodes in relation to the vehicle's capacity.
    normalized_distance_matrix = distance_matrix / demands.unsqueeze(1)
    
    # Normalize the demands to create a demand vector that can be used to scale the distance matrix.
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum
    
    # Calculate the heuristic value for each edge by multiplying the normalized distance
    # with the normalized demand for each customer.
    heuristic_matrix = normalized_distance_matrix * normalized_demands.unsqueeze(0)
    
    # Subtract the demand of the destination node from the heuristic value to penalize
    # edges that would lead to exceeding the vehicle's capacity.
    heuristic_matrix -= demands.unsqueeze(1)
    
    return heuristic_matrix