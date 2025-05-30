import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the input is in the correct shape and type
    if not (isinstance(distance_matrix, torch.Tensor) and isinstance(demands, torch.Tensor)):
        raise ValueError("Both distance_matrix and demands should be torch.Tensor objects.")
    
    # Normalize demands by the total vehicle capacity
    vehicle_capacity = demands.sum() / len(demands)
    normalized_demands = demands / vehicle_capacity
    
    # Initialize the heuristic matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Apply Inverse Distance Heuristic (IDH) for the heuristic matrix
    # We use a simple inverse distance as the heuristic value
    # In a real-world scenario, this should be combined with demand normalization
    heuristics = 1.0 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Incorporate a demand-sensitive penalty mechanism
    # Increase the heuristic value for edges that lead to overloading
    for i in range(len(demands)):
        for j in range(len(demands)):
            if i != j:
                # If adding this customer to the vehicle would cause it to overload
                if normalized_demands[i] + normalized_demands[j] > 1.0:
                    heuristics[i, j] *= 1.5  # Increase the penalty by 50%
    
    return heuristics