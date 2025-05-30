import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Initialize a tensor to store the heuristics with the same shape as the distance matrix
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristics for each edge
    # We use a simple heuristic that considers the normalized demand of the destination node
    heuristics = heuristics - normalized_demands
    
    # Optionally, you can include additional heuristics here
    # For example, you could add a term that encourages visiting nodes with higher demands first
    
    return heuristics