import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative sum of demands, subtracting from the total demand
    # to get the remaining capacity at each node
    remaining_capacity = 1.0 - demands.cumsum(dim=0)
    
    # Calculate the cumulative sum of the negative of the distance matrix
    # to get the "accumulated cost" to visit nodes in order
    accumulated_cost = -distance_matrix.cumsum(dim=0)
    
    # Combine the remaining capacity at each node with the accumulated cost
    # to determine the "cost of adding each edge"
    cost_of_adding_edge = accumulated_cost + remaining_capacity
    
    # We use the absolute value of the difference in capacity at the nodes at either end of each edge
    # to represent the potential gain if we were to add that edge
    edge_potential_gain = torch.abs(remaining_capacity[1:] - remaining_capacity[:-1])
    
    # Combine the potential gain and cost of adding an edge to create a heuristic value
    heuristic_values = cost_of_adding_edge[1:] - cost_of_adding_edge[:-1] + edge_potential_gain
    
    # Set the heuristic values for the edges leaving the depot (i.e., to the first customer)
    # and the edges entering the depot (i.e., from the last customer) to a negative value
    # to discourage visiting them unless absolutely necessary
    heuristic_values[0] = -heuristic_values[0]
    heuristic_values[-1] = -heuristic_values[-1]
    
    # Replace negative values with 0, since they represent undesirable edges
    heuristic_values = torch.clamp(heuristic_values, min=0)
    
    return heuristic_values