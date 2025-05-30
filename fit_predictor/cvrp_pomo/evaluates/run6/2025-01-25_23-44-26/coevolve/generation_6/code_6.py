import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands
    normalized_demands = demands / demands.sum()
    
    # Inverse Distance Heuristic (IDH): Assign customers to vehicles based on the reciprocal of their distance
    # Initialize the heuristic matrix with large negative values
    heuristic_matrix = -torch.ones_like(distance_matrix)
    
    # Set the heuristic values based on the reciprocal of the distance
    heuristic_matrix[distance_matrix != 0] = 1 / distance_matrix[distance_matrix != 0]
    
    # Demand penalty function: Increase the cost of assigning high-demand customers to vehicles near their capacity limits
    # We will add a penalty to the heuristic values if the sum of demands assigned to a vehicle is close to the capacity
    vehicle_capacity = 1.0  # Example capacity, can be adjusted as needed
    demand_penalty_threshold = vehicle_capacity * 0.8  # Threshold for applying penalty
    
    # Iterate over each vehicle and apply the demand penalty
    for i in range(1, len(demands) + 1):
        demand_sum = demands[:i].sum()
        if demand_sum > vehicle_capacity:
            # Calculate the penalty based on how close the sum of demands is to the capacity
            penalty = (demand_sum - vehicle_capacity) / demand_sum
            # Apply the penalty to the heuristic values for this vehicle
            heuristic_matrix[:i, i:] += penalty
    
    return heuristic_matrix