import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Calculate the average demand per vehicle
    average_demand = total_demand / distance_matrix.shape[0]
    
    # Create a matrix of the same shape as the distance matrix initialized with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Iterate over the distance matrix to calculate the heuristics
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            # If the edge is not the diagonal (i.e., not the depot to itself)
            if i != j:
                # Calculate the potential demand if this edge is included
                potential_demand = demands[i] + demands[j]
                
                # If the potential demand is less than or equal to the average demand
                if potential_demand <= average_demand:
                    # Calculate the heuristic value
                    heuristic_value = (potential_demand / average_demand) - 1
                    
                    # Assign the heuristic value to the corresponding edge
                    heuristics_matrix[i, j] = heuristic_value
                else:
                    # If the potential demand exceeds the average demand, mark as undesirable
                    heuristics_matrix[i, j] = -1
    
    return heuristics_matrix