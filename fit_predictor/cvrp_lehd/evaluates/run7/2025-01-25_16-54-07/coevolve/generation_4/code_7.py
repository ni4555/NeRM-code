import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    depot = 0  # depot is indexed by 0
    # Initialize a matrix of the same shape as the distance matrix with all positive values
    heuristic_matrix = torch.full_like(distance_matrix, fill_value=1.0)
    # Calculate the cumulative demand matrix
    cumulative_demand = torch.cumsum(demands[1:], dim=0)  # ignore the demand of the depot
    cumulative_demand = torch.cat((torch.zeros(1), cumulative_demand))  # add the depot demand

    # Nearest neighbor heuristic: calculate the cost of going from each node to the nearest customer
    for i in range(1, n):
        distances_to_customer = distance_matrix[i]
        min_distance = torch.min(distances_to_customer[distances_to_customer > 0])
        heuristic_matrix[i] = -min_distance

    # Demand-driven route optimization phase: adjust paths based on real-time demand fluctuations
    for i in range(1, n):
        for j in range(1, n):
            if distances_to_customer[j] > 0 and cumulative_demand[i] + demands[j] <= 1:
                # Check if adding this edge is feasible without overloading the vehicle
                heuristic_matrix[i, j] = -distances_to_customer[j] + (1 - cumulative_demand[i] - demands[j])

    return heuristic_matrix