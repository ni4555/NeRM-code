import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands tensor is a column vector
    demands = demands.view(-1, 1)
    
    # Calculate the distance from the depot to each customer and back to the depot
    distance_to_customer = distance_matrix[0, :]
    distance_from_customer_to_depot = distance_matrix[:, 0]
    
    # Calculate the total distance for each edge (including the return to the depot)
    total_distance = distance_to_customer + distance_from_customer_to_depot
    
    # Calculate the heuristic value as a product of total distance and demand
    heuristic_values = total_distance * demands
    
    # Subtract the maximum heuristic value from all to ensure non-negative values
    max_heuristic_value = torch.max(heuristic_values)
    heuristic_values -= max_heuristic_value
    
    return heuristic_values