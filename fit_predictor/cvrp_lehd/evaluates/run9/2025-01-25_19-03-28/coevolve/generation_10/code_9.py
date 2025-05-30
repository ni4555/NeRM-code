import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()
    
    # Normalize demands by the vehicle capacity
    normalized_demands = demands / vehicle_capacity
    
    # Calculate the total distance for each customer
    total_distance = distance_matrix.sum(dim=1)
    
    # Calculate the load balance score for each customer
    load_balance_score = (normalized_demands * total_distance).sum(dim=1)
    
    # Calculate the service time score for each customer
    service_time_score = torch.log1p(total_distance)  # Using log to give higher scores to longer distances
    
    # Combine the scores into a single heuristic matrix
    heuristic_matrix = load_balance_score + service_time_score
    
    # Adjust the heuristic matrix to have negative values for undesirable edges and positive values for promising ones
    # We use a simple threshold to convert the scores to heuristics
    threshold = heuristic_matrix.max()
    heuristic_matrix = torch.where(heuristic_matrix > threshold, -torch.ones_like(heuristic_matrix), heuristic_matrix)
    
    return heuristic_matrix