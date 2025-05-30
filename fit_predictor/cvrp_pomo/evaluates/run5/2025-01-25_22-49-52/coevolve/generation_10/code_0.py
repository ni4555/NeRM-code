import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize the distance matrix to account for vehicle capacity
    normalized_distance_matrix = distance_matrix / demands.unsqueeze(1)
    
    # Define a constant for the demand relaxation factor
    DEMAND_RELAXATION_FACTOR = 0.1
    
    # Relax demands to handle dynamic changes
    relaxed_demands = demands * (1 + DEMAND_RELAXATION_FACTOR)
    
    # Calculate the initial heuristic values based on normalized distances
    initial_heuristic = -normalized_distance_matrix
    
    # Incorporate demand relaxation into the heuristic
    demand_based_heuristic = initial_heuristic + relaxed_demands
    
    # Apply node partitioning to optimize path decomposition
    # A simple approach could be to use customer demands as weights for partitioning
    # Here we use the average of the relaxed demand as a proxy for partitioning
    average_relaxed_demand = relaxed_demands.mean()
    partitioning_heuristic = torch.where(demand_based_heuristic > average_relaxed_demand,
                                        demand_based_heuristic,
                                        average_relaxed_demand)
    
    # Apply a dynamic window approach to adjust the heuristic values in real-time
    # This is a placeholder for the dynamic window logic, which would be more complex
    # and would depend on the specific real-time changes and their timing.
    # For simplicity, we'll just add a small positive value to simulate this effect.
    dynamic_window_adjustment = torch.ones_like(partitioning_heuristic) * 0.05
    dynamic_window_heuristic = partitioning_heuristic + dynamic_window_adjustment
    
    return dynamic_window_heuristic