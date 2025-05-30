import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Node partitioning: create a partitioning matrix based on demand threshold
    threshold = 0.1  # Example threshold for partitioning
    partitioning_matrix = (demands > threshold).to(torch.float32)
    
    # Demand relaxation: relax demands within the partition to balance vehicle loads
    relaxed_demands = torch.where(partitioning_matrix, demands * 0.9, demands)  # Example relaxation
    
    # Path decomposition: create a matrix of potential paths based on partitioning
    num_nodes = distance_matrix.shape[0]
    path_potential = torch.zeros_like(distance_matrix)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # Example: increase potential if the demand is high and the distance is short
                path_potential[i, j] = relaxed_demands[i] * relaxed_demands[j] * (1 - distance_matrix[i, j])
    
    # Dynamic window approach: adjust potential based on distance matrix
    # Example: reduce potential if the path distance is too long
    distance_threshold = 10  # Example threshold for distance
    path_potential = torch.where(distance_matrix < distance_threshold, path_potential, path_potential * 0.5)
    
    # Return the heuristics matrix
    return path_potential