import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative sum of demands to find the first node where the demand exceeds the vehicle capacity
    demand_cumsum = torch.cumsum(demands, dim=0)
    
    # Find the index of the first node where the demand exceeds the vehicle capacity
    capacity_exceeded_index = torch.where(demand_cumsum > 1.0)[0]
    
    # Initialize the heuristic matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # For each node, calculate the heuristic value
    for i in range(len(capacity_exceeded_index) - 1):
        start_index = capacity_exceeded_index[i]
        end_index = capacity_exceeded_index[i + 1]
        
        # Calculate the heuristic value for edges from start_index to end_index - 1
        for j in range(start_index, end_index):
            heuristics[j, start_index] = -1.0
            heuristics[start_index, j] = -1.0
            
            # Calculate the heuristic value for the edge from end_index - 1 to start_index
            edge_heuristic = distance_matrix[end_index - 1, start_index] - distance_matrix[j, start_index]
            heuristics[end_index - 1, start_index] = edge_heuristic
            heuristics[start_index, end_index - 1] = edge_heuristic
    
    # For the last segment of the route, handle the edge from the last capacity exceeded node back to the depot
    last_index = capacity_exceeded_index[-1]
    edge_heuristic = distance_matrix[last_index, 0] - distance_matrix[last_index, capacity_exceeded_index[-2]]
    heuristics[last_index, 0] = edge_heuristic
    heuristics[0, last_index] = edge_heuristic
    
    return heuristics