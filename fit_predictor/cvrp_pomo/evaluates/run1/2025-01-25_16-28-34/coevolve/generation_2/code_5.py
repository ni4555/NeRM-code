import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Compute the normalized demands to be used for heuristics
    normalized_demands = demands / demands.sum()
    
    # Calculate the negative distance as a heuristic measure
    # The smaller the distance, the more promising the edge
    # In the CVRP, we want to avoid long distances, so we use negative values
    negative_distance_matrix = -distance_matrix
    
    # Incorporate demand into the heuristic to prioritize edges with lower demands
    # Lower demand means less capacity is needed, making the route more promising
    demand_adjustment = normalized_demands[:, None] * normalized_demands[None, :]
    heuristic_matrix = negative_distance_matrix + demand_adjustment
    
    return heuristic_matrix