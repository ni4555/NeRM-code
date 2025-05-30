import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the savings for each edge
    savings = 2 * (distance_matrix ** 2) - (distance_matrix.sum(dim=1) ** 2) - (distance_matrix.sum(dim=0) ** 2)
    
    # Incorporate normalized demand into savings
    savings = savings - normalized_demands.unsqueeze(1) * distance_matrix
    
    # Apply a penalty for edges that are part of the same route (self-loops)
    savings = savings - (distance_matrix < 1e-6) * 1e6
    
    # Apply a positive heuristics for edges with savings
    heuristics = savings * (savings > 0)
    
    return heuristics