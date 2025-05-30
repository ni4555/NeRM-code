import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Node partitioning: Group nodes based on their demand and distance
    demand_partition = torch.argsort(demands)
    distance_partition = torch.argsort(distance_matrix[0])

    # Demand relaxation: Adjust demands to be multiples of vehicle capacity
    vehicle_capacity = 1.0  # Assuming unit capacity for simplicity
    relaxed_demands = demands / vehicle_capacity
    relaxed_demands = relaxed_demands.floor().mul(vehicle_capacity).ceil()

    # Path decomposition: Calculate the potential of each path
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate the total demand on the path if it were to be taken
                path_demand = relaxed_demands[i] + relaxed_demands[j]
                # Calculate the potential score based on the total demand and distance
                potential_score = -distance_matrix[i, j] + path_demand
                # Update the heuristic matrix
                heuristic_matrix[i, j] = potential_score

    return heuristic_matrix