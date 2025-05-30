import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity by summing the demands (excluding the depot)
    vehicle_capacity = demands.sum()
    
    # Normalize the demands vector by the vehicle capacity
    normalized_demands = demands / vehicle_capacity
    
    # Calculate the difference between each customer's demand and the vehicle capacity
    # This will be used to determine the heuristics values
    demand_diff = normalized_demands - 1
    
    # The heuristic value for each edge will be a function of the distance and the demand difference
    # We use the distance as a cost and the demand difference as a negative incentive to visit nodes with high demand
    # This encourages a load distribution that balances the load across vehicles
    heuristics_values = -distance_matrix + demand_diff
    
    # The heuristics function should return negative values for undesirable edges
    # and positive values for promising ones. We can do this by setting a threshold
    # based on the minimum distance in the matrix (assuming that very short distances are more desirable).
    min_distance = torch.min(distance_matrix)
    threshold = min_distance * 0.5  # Threshold can be adjusted based on problem specifics
    
    # Apply the threshold to create the heuristics matrix
    heuristics = torch.where(heuristics_values < threshold, heuristics_values, torch.zeros_like(heuristics_values))
    
    return heuristics