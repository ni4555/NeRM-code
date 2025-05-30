import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity as the sum of customer demands
    total_capacity = demands.sum()

    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity

    # Normalize the distance matrix by the maximum distance in the matrix
    max_distance = distance_matrix.max()
    normalized_distance_matrix = distance_matrix / max_distance

    # Calculate the potential values for explicit depot handling
    # This could be a simple heuristic like the demand of the depot node
    depot_demand = normalized_demands[0]
    
    # Initialize the heuristics matrix with negative values
    heuristics_matrix = -normalized_distance_matrix

    # Adjust the heuristics for edges to the depot based on the demand
    heuristics_matrix[:, 0] += depot_demand
    heuristics_matrix[0, :] += depot_demand

    # Integrate constraint programming by considering the capacities
    # Here we use a simple heuristic where we add a positive value for each customer
    # that does not exceed the capacity of the vehicle (which is 1 in this case)
    for i in range(1, len(demands)):
        if normalized_demands[i] <= 1:
            heuristics_matrix[i, :] += 1
            heuristics_matrix[:, i] += 1

    # Integrate dynamic window approach by considering the dynamic changes
    # For simplicity, we can add a bonus for short edges
    short_edge_bonus = 0.1
    heuristics_matrix[distance_matrix < short_edge_bonus * max_distance] += short_edge_bonus

    return heuristics_matrix