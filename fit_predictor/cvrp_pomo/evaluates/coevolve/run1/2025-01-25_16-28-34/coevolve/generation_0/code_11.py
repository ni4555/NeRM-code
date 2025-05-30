import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum demand per vehicle (assuming all vehicles have the same capacity)
    max_demand_per_vehicle = 1.0
    
    # Calculate the sum of demands for each node (including the depot)
    node_demand_sum = torch.sum(demands)
    
    # Calculate the sum of distances for each possible route (including the depot to itself)
    distance_sum = torch.sum(distance_matrix)
    
    # Normalize the sum of distances by the total number of nodes to get an average distance
    average_distance = distance_sum / len(distance_matrix)
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # For each edge in the distance matrix, calculate the heuristics value
    # We use the following heuristic:
    # - For each edge, calculate the total distance for a round trip
    # - Divide this by the sum of demands to get an average demand per distance unit
    # - If the average demand per distance unit is greater than the max demand per vehicle,
    #   the edge is marked as undesirable (negative heuristic value)
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j:
                round_trip_distance = 2 * distance_matrix[i, j]
                average_demand_per_distance_unit = (demands[i] + demands[j]) / round_trip_distance
                if average_demand_per_distance_unit > max_demand_per_vehicle:
                    heuristics[i, j] = -1.0
                else:
                    heuristics[i, j] = 1.0
    
    return heuristics