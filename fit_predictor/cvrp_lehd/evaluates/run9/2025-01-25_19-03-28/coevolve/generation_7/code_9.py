import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand for the depot node (index 0)
    depot_demand = demands[0]
    
    # Calculate the total vehicle capacity (assuming demands are normalized)
    total_capacity = demands.sum()
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Iterate over each edge in the distance matrix
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            # Calculate the potential cost for the edge
            edge_cost = distance_matrix[i, j]
            
            # If it's the depot to a customer or a customer to the depot
            if i == 0 or j == 0:
                # Calculate the contribution to the demand balance
                contribution = demands[j] - demands[i]
                
                # If it's a depot to customer edge and the demand is positive
                if i == 0 and contribution > 0:
                    heuristics[i, j] = contribution / total_capacity
                # If it's a customer to depot edge and the demand is negative
                elif j == 0 and contribution < 0:
                    heuristics[i, j] = -contribution / total_capacity
            else:
                # For customer to customer edges, consider both demands
                contribution = demands[j] - demands[i]
                
                # If both contributions are positive, calculate the sum
                if contribution > 0:
                    heuristics[i, j] = contribution / total_capacity
    
    return heuristics