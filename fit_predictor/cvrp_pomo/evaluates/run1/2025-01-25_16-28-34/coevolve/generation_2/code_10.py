import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to be between 0 and 1
    demands_normalized = demands / demands.sum()
    
    # Calculate the heuristics based on the normalized demands
    # Using a simple heuristic: the demand of the customer
    heuristics = demands_normalized
    
    # To make the heuristic more effective, we can add a term that penalizes
    # edges that lead to a customer with a high demand relative to the total capacity
    # This is a simple heuristic that assumes we want to avoid high-demand customers
    # if possible, to maintain load balance
    heuristics += (1 - demands_normalized) * 0.5
    
    # We can also add a term that rewards edges that lead to customers with lower
    # demand, which can help in the exploration of the solution space
    heuristics += demands_normalized * 0.5
    
    # Ensure that the heuristics are negative for undesirable edges and positive for
    # promising ones by subtracting the maximum value from the heuristics
    heuristics -= heuristics.max()
    
    return heuristics