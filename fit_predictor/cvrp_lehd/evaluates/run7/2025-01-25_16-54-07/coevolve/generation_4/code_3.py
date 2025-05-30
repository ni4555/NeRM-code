import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize a tensor with the same shape as the distance matrix filled with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Nearest neighbor heuristic: assign each node to its nearest neighbor
    for i in range(1, n):
        # Find the index of the nearest neighbor for node i
        nearest_neighbor = torch.argmin(distance_matrix[i])
        # Set the heuristic value for the edge from i to its nearest neighbor
        heuristics[i, nearest_neighbor] = -1  # Mark as undesirable
    
    # Demand-driven route optimization: adjust the heuristic values based on demand
    for i in range(1, n):
        # Calculate the cumulative demand along the path from the depot to node i
        cumulative_demand = demands[i]
        for j in range(1, i):
            cumulative_demand += demands[j]
            # If the cumulative demand exceeds the vehicle capacity, mark the edge as undesirable
            if cumulative_demand > 1.0:
                heuristics[j, i] = -1
                break
    
    # Mark the edges from the depot to all nodes as promising
    heuristics[:, 0] = 1.0
    heuristics[0, :] = 1.0
    
    return heuristics