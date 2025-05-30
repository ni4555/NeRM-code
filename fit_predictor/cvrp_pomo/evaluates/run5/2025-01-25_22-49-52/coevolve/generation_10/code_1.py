import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Extract the size of the distance matrix
    n = distance_matrix.size(0)
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Define the weight for demand relaxation
    demand_weight = 0.5
    
    # Demand relaxation for dynamic changes in node demands
    relaxed_demands = demands / (1 + demand_weight)
    
    # Node partitioning for path decomposition
    partition_threshold = 1.0  # This is an example threshold
    demand_threshold = 0.5     # This is an example threshold
    
    # Iterate over all edges
    for i in range(n):
        for j in range(1, n):
            # Calculate the heuristic based on distance and relaxed demand
            edge_heuristic = distance_matrix[i, j] - relaxed_demands[i] - relaxed_demands[j]
            
            # Apply demand threshold to filter out edges with high demand
            if relaxed_demands[i] + relaxed_demands[j] > demand_threshold:
                edge_heuristic *= 0.1
            
            # Apply partition threshold to optimize path decomposition
            if i < partition_threshold or j < partition_threshold:
                edge_heuristic *= 0.9
            
            # Store the heuristic value in the matrix
            heuristic_matrix[i, j] = edge_heuristic
    
    return heuristic_matrix