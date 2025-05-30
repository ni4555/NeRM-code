import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand for each node
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Iterate over all edges
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # If the destination node demand is greater than the cumulative demand
                # from the starting node to the current node, mark this edge as undesirable
                if cumulative_demand[j] - cumulative_demand[i] > 1:
                    heuristics[i, j] = -1
    
    return heuristics