import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix to ensure no distance is zero
    distance_matrix = distance_matrix - torch.min(distance_matrix)
    
    # Calculate normalized demand
    normalized_demands = demands / torch.sum(demands)
    
    # Define the demand penalty function
    def demand_penalty(demand):
        # High demand customers close to capacity have a higher penalty
        return demand * torch.clamp((1 - demand), min=0, max=1)
    
    # Apply the demand penalty function to the demand vector
    demand_penalty_vector = torch.vectorized(demand_penalty)(normalized_demands)
    
    # Compute the heuristic values based on the inverse distance and demand penalty
    heuristic_matrix = (1 / distance_matrix) * demand_penalty_vector
    
    # Ensure the heuristic values are negative for undesirable edges
    heuristic_matrix[distance_matrix == 0] = float('-inf')
    
    return heuristic_matrix