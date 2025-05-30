import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand for each customer
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Calculate the cumulative distance for each edge in the distance matrix
    # We use torch.triu to get the upper triangle of the distance matrix
    # which represents the distances from the depot to all other nodes
    cumulative_distance = torch.cumsum(distance_matrix[torch.triu_indices(distance_matrix.shape[0], 1)], dim=1)
    
    # Calculate the potential cost of visiting a customer at each step
    # This is the sum of the distance to the customer and the fraction of the demand
    # that has not been covered yet
    potential_cost = cumulative_distance + (cumulative_demand * demands)
    
    # To promote visiting customers that are both close and have high demand remaining,
    # we can use the negative of the potential cost as the heuristic value
    # This encourages the genetic algorithm to prioritize these edges
    heuristic_values = -potential_cost
    
    # Fill in the diagonal of the distance matrix with a large negative value
    # to avoid considering the depot as a customer to be visited
    torch.fill_diagonal_(heuristic_values, float('-inf'))
    
    return heuristic_values