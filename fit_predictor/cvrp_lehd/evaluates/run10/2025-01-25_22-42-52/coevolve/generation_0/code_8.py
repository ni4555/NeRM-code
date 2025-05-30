import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the distance_matrix and demands are in the same shape
    assert distance_matrix.shape == demands.shape, "Distance matrix and demands must have the same shape."
    
    # Calculate the sum of demands
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity (assuming total_demand is the capacity)
    normalized_demands = demands / total_demand
    
    # Compute the heuristics based on the distance and normalized demand
    # We can use a simple heuristic like the negative of the distance
    # This can be adjusted with a function of the demand as well
    heuristics = -distance_matrix + normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Replace negative values with a very small positive value to avoid taking the log of zero
    heuristics = torch.clamp(heuristics, min=1e-8)
    
    return heuristics