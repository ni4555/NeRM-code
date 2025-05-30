import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized by the vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the maximum feasible demand that can be carried by a vehicle
    max_demand = demands.max()
    
    # Initialize a matrix with zeros of the same shape as the distance matrix
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Loop over all possible combinations of edges (i, j) where i is not equal to j
    for i in range(1, len(distance_matrix)):
        for j in range(1, len(distance_matrix)):
            # Calculate the demand for the edge (i, j)
            edge_demand = demands[i] + demands[j]
            
            # If the edge demand is less than or equal to the maximum feasible demand,
            # assign a positive heuristic value, otherwise assign a negative value
            if edge_demand <= max_demand:
                heuristics_matrix[i, j] = torch.exp(-distance_matrix[i, j])
                heuristics_matrix[j, i] = heuristics_matrix[i, j]
            else:
                heuristics_matrix[i, j] = -float('inf')
                heuristics_matrix[j, i] = heuristics_matrix[i, j]
    
    # Handle the diagonal edges (self-loops) by setting their heuristic to -infinity
    heuristics_matrix[:, 0] = -float('inf')
    heuristics_matrix[0, :] = -float('inf')
    
    # Normalize the heuristic matrix to have a sum of 1 across each row (representing a vehicle)
    heuristics_matrix = (heuristics_matrix + heuristics_matrix.t()) / 2
    heuristics_matrix = heuristics_matrix / heuristics_matrix.sum(dim=1, keepdim=True)
    
    return heuristics_matrix