import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands to the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristics based on normalized demands
    # Here we use a simple heuristic where the attractiveness of an edge is inversely proportional to the demand
    heuristics = -normalized_demands
    
    # Apply a penalty for edges that lead to a higher total demand that exceeds the vehicle capacity
    # This is a simplistic approach and might need to be adjusted based on the specific problem instance
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Skip the depot node
                # Calculate the total demand if this edge is included
                total_demand = demands[i] + demands[j]
                # If the total demand exceeds the capacity, apply a penalty
                if total_demand > total_capacity:
                    heuristics[i, j] = -float('inf')
    
    return heuristics