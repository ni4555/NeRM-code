import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity as the sum of all customer demands
    total_capacity = demands.sum()
    
    # Normalize the customer demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values for each edge
    # A simple heuristic could be to use the negative of the distance (if distance is 0, use -1)
    # This is a simplistic heuristic and might not be suitable for complex VRP scenarios
    heuristics_matrix = -distance_matrix
    
    # Adjust for normalized demands, giving more weight to edges with higher demand
    heuristics_matrix += (normalized_demands[:, None] * distance_matrix)
    
    # Ensure that edges to the depot (index 0) have a high priority
    heuristics_matrix[:, 0] = -1e6
    heuristics_matrix[0, :] = -1e6
    
    return heuristics_matrix