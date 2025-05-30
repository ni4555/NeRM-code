import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Initialize a matrix to store heuristic values with the same shape as the distance matrix
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Compute the maximum distance for each row
    max_distance_per_row, _ = torch.max(distance_matrix, dim=1)
    
    # Compute the sum of demands for each row
    sum_demands_per_row = torch.sum(demands)
    
    # For each row, compute the heuristic as -max_distance if demand is met,
    # otherwise add a small positive value for edges that cannot be traversed
    # based on capacity.
    for i in range(n):
        # Demand for the current node is the demand for the node itself
        node_demand = demands[i]
        
        # Check if the current node demand is less than or equal to the remaining capacity
        # If it is, the edge is promising, so assign it a negative value (more negative means more promising)
        if node_demand <= sum_demands_per_row - node_demand:
            heuristics_matrix[i] = -max_distance_per_row[i]
        else:
            # If not, it's not possible to visit this node given the capacity, so assign a positive value
            heuristics_matrix[i] = torch.full_like(max_distance_per_row[i], torch.tensor(0.1))
    
    return heuristics_matrix