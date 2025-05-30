import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize them
    total_demand = demands.sum()
    
    # Normalize demands
    normalized_demands = demands / total_demand
    
    # Compute the potential heuristics based on normalized demands and distance
    # The heuristic function can be designed in various ways. Here, a simple
    # example is used where the heuristic is the inverse of the normalized demand
    # multiplied by the distance. This assumes that closer and higher-demand nodes
    # are more promising to visit.
    heuristics = normalized_demands * (1 / distance_matrix)
    
    # Clip the values to ensure no division by zero and to avoid numerical issues
    heuristics = torch.clamp(heuristics, min=0.0001)
    
    return heuristics