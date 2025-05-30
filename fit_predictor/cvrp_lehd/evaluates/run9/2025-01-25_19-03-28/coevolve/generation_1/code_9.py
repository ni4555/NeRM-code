import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize them
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Compute the heuristics using a simple formula that considers the normalized demand
    # and the distance. For example, we can use the product of normalized demand and distance.
    # Negative values are undesirable edges, positive values are promising ones.
    heuristics = normalized_demands * distance_matrix
    
    return heuristics