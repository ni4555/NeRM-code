import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize a tensor with zeros, same shape as distance_matrix
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Calculate the heuristic value for each edge
    # The heuristic is designed to be positive for promising edges
    # and negative for undesirable edges.
    # Here we use a simple heuristic that considers the distance and demand
    # We want to prioritize edges that have lower distance and lower demand.
    for i in range(n):
        for j in range(n):
            if i != j:
                # If the edge is between the depot and a customer
                if i == 0 and demands[j] > 0:
                    # Reward the edge if it has lower distance and demand
                    heuristic_matrix[i, j] = -distance_matrix[i, j] - demands[j]
                # If the edge is between two customers
                elif demands[i] > 0 and demands[j] > 0:
                    # Reward the edge if it has lower distance
                    heuristic_matrix[i, j] = -distance_matrix[i, j]
    
    return heuristic_matrix