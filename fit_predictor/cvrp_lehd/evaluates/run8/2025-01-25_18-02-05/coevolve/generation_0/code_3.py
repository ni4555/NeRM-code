import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # The number of nodes (including the depot)
    num_nodes = distance_matrix.shape[0]
    
    # Normalize demands by the total vehicle capacity for simplicity
    # Assuming that the total vehicle capacity is 1 for normalization
    total_demand = demands.sum()
    normalized_demands = demands / total_demand
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Iterate over all edges (i, j) except for the diagonal
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # Calculate the heuristic value
                # We assume that the more demanding the customer, the higher the heuristic
                # This is a simple heuristic where we add the normalized demand
                heuristics[i, j] = normalized_demands[j]
                
    return heuristics