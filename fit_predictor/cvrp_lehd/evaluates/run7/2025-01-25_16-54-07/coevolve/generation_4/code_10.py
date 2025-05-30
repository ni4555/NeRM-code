import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming the distance matrix is symmetric and the demands are normalized by total vehicle capacity
    
    # Initialize a matrix with zeros to store heuristic values
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the cumulative demand at each node
    cumulative_demand = demands.cumsum(0)
    
    # Find the nearest neighbor for each node starting from the depot (node 0)
    nearest_neighbors = torch.argmin(distance_matrix[:, 1:], dim=1) + 1  # +1 to adjust for 0-indexed depot
    
    # Calculate the initial heuristic values based on the nearest neighbor
    for i in range(1, len(nearest_neighbors)):
        heuristic_matrix[i, nearest_neighbors[i]] = -1  # Unpromising edge back to the depot
        
    # Adjust the heuristic values based on cumulative demand
    for i in range(1, len(cumulative_demand)):
        for j in range(1, len(nearest_neighbors)):
            if cumulative_demand[i] > demands[j] and cumulative_demand[i] - demands[j] <= demands[j]:
                # If the demand at node i can be covered by the demand at node j
                heuristic_matrix[i, j] = -1  # Unpromising edge if it exceeds capacity
            else:
                # Promote edges that are part of a potential feasible route
                if distance_matrix[i, nearest_neighbors[i]] > distance_matrix[i, j]:
                    heuristic_matrix[i, j] = 1  # Promising edge to a closer neighbor
    
    return heuristic_matrix