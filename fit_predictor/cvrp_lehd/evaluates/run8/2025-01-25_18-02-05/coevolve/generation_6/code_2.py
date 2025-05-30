import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming demands are normalized by the total vehicle capacity and the depot node is at index 0
    # We want to penalize edges that lead to a customer with a high demand relative to the distance
    # This heuristic uses a negative value for undesirable edges and positive for promising ones
    
    # Calculate the relative demand by dividing each customer demand by the sum of all demands
    total_demand = demands.sum()
    relative_demand = demands / total_demand
    
    # Calculate a penalty factor for each edge based on the relative demand and distance
    # We want to penalize edges that lead to customers with high relative demand at long distances
    penalty_factor = relative_demand * distance_matrix
    
    # The heuristic value is the negative of the penalty factor
    # Negative values will indicate undesirable edges, positive values will indicate promising ones
    heuristic_values = -penalty_factor
    
    return heuristic_values