import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands to be between 0 and 1
    normalized_demands = demands / demands.sum()
    
    # Compute the heuristic values for each edge
    # The heuristic is a combination of the demand and the distance, adjusted by a penalty for high demands
    # Here, we use a simple heuristic: demand * distance
    # We can adjust the heuristic function to better suit the problem's specifics
    
    # The heuristic is negative for undesirable edges and positive for promising ones
    # The idea is to encourage routes with lower demand and shorter distances
    heuristic_values = distance_matrix * normalized_demands
    
    return heuristic_values