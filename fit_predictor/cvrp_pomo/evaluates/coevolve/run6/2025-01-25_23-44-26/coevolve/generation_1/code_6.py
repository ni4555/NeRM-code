import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum possible load for a route (total capacity minus the depot demand)
    max_load = demands.sum() - demands[0]
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Iterate over all pairs of nodes (except the depot with itself)
    for i in range(1, len(demands)):
        for j in range(1, len(demands)):
            if i != j:
                # Calculate the contribution to the load if the edge from i to j is included
                edge_load = demands[i] + demands[j]
                
                # If the edge load is less than the vehicle's capacity, it's a promising edge
                if edge_load <= max_load:
                    # The heuristic value is the negative of the distance (to encourage shortest paths)
                    heuristics[i, j] = -distance_matrix[i, j]
    
    return heuristics