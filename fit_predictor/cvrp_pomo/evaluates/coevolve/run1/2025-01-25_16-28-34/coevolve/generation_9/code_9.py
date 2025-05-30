import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands to get the fraction of the capacity each customer represents
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values for each edge
    # The heuristic is designed to be negative for edges that would exceed vehicle capacity
    # and positive for edges that would not exceed vehicle capacity
    for i in range(distance_matrix.size(0)):
        for j in range(distance_matrix.size(1)):
            if i == j:
                # No cost for the depot to itself
                heuristic_matrix[i, j] = 0
            else:
                # Calculate the potential load if this edge is included
                potential_load = normalized_demands[i] + normalized_demands[j]
                # If the potential load exceeds the capacity, the heuristic is negative
                if potential_load > 1.0:
                    heuristic_matrix[i, j] = -torch.clamp(potential_load - 1.0, min=-1e6, max=0)
                else:
                    # Otherwise, the heuristic is positive
                    heuristic_matrix[i, j] = torch.clamp(potential_load - 1.0, min=0, max=1e6)
    
    return heuristic_matrix