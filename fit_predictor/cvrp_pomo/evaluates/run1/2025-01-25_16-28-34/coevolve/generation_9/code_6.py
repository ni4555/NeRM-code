import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands to the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the potential load for each edge
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j:  # Skip the depot node
                # Calculate the potential load if this edge is included
                potential_load = normalized_demands[i] + normalized_demands[j]
                
                # If the potential load is within the vehicle capacity, assign a positive heuristic value
                if potential_load <= 1.0:
                    heuristics[i, j] = 1 - potential_load
                else:
                    # If the potential load exceeds the vehicle capacity, assign a negative heuristic value
                    heuristics[i, j] = potential_load - 1.0
    
    return heuristics