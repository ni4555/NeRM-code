import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming distance_matrix and demands are both 1D tensors where the depot node is at index 0
    # and the demands are normalized by the total vehicle capacity.
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Demand relaxation: Calculate the total demand
    total_demand = demands.sum()
    
    # Node partitioning: Group nodes by demand levels
    demand_levels = torch.unique(demands)
    
    # Path decomposition: Start with the depot and explore nodes based on their demand
    for level in demand_levels:
        # Filter nodes that have the current demand level
        nodes_with_demand = (demands == level)
        
        # For each node with the current demand level, calculate the heuristic
        for node in torch.nonzero(nodes_with_demand):
            # Calculate the heuristic as the negative of the distance to the depot
            # and adjust it based on the demand relaxation
            heuristic_matrix[node, 0] = -distance_matrix[node, 0] + (level / total_demand)
    
    return heuristic_matrix