import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the cumulative demand for each edge
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Apply the nearest neighbor heuristic
    for i in range(1, len(demands)):
        min_distance = torch.min(distance_matrix[i], dim=0).values
        heuristic_matrix[i] = -min_distance
    
    # Incorporate cumulative demand checks and capacity constraints
    for i in range(1, len(demands)):
        for j in range(1, len(demands)):
            if i != j:
                if cumulative_demand[i] > 1:  # Check if the current vehicle is overloaded
                    heuristic_matrix[i, j] = -float('inf')  # Mark this edge as undesirable
                else:
                    # Adjust the heuristic based on real-time demand fluctuations
                    if cumulative_demand[j] - cumulative_demand[i] < 1:
                        heuristic_matrix[i, j] += 1
    
    return heuristic_matrix