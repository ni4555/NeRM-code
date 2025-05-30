import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()
    
    # Normalize demands by vehicle capacity
    normalized_demands = demands / vehicle_capacity
    
    # Initialize the potential value matrix
    potential_value_matrix = torch.zeros_like(distance_matrix)
    
    # Node partitioning: Identify the nodes with the highest demand
    high_demand_nodes = normalized_demands > 0.1  # threshold can be adjusted
    
    # Demand relaxation: Reduce the demand for high-demand nodes
    relaxed_demands = torch.where(high_demand_nodes, demands * 0.9, demands)
    
    # Path decomposition: Compute the potential value for each edge
    for i in range(distance_matrix.size(0)):
        for j in range(distance_matrix.size(1)):
            if i != j:  # Exclude the depot node from the calculation
                # Calculate the potential value for edge (i, j)
                potential_value = relaxed_demands[i] + relaxed_demands[j] - demands[i] - demands[j]
                # Update the potential value matrix
                potential_value_matrix[i, j] = potential_value
    
    # Apply an explicit potential value heuristic for direct potential value calculation
    # The heuristic assigns negative values to undesirable edges and positive values to promising ones
    # Here we use a simple heuristic where we subtract the potential value from the distance
    heuristic_matrix = distance_matrix - potential_value_matrix
    
    return heuristic_matrix