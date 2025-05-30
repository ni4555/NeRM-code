import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize demands
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic values for each edge
    # Here, we use a simple heuristic: the negative of the distance multiplied by the normalized demand
    # This heuristic assumes that closer nodes with higher demand are more promising
    heuristics = -distance_matrix * normalized_demands
    
    return heuristics