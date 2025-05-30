import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand of each edge
    cumulative_demand = demands.cumsum(dim=0)
    
    # Calculate the number of customers that would be visited if the edge is included
    num_customers = torch.arange(1, distance_matrix.shape[0], dtype=torch.float32)
    
    # Calculate the potential improvement in the number of customers if the edge is included
    potential_improvement = num_customers - cumulative_demand
    
    # Calculate the heuristic values
    # We want to promote edges that have a higher potential improvement
    # and are not at the end of a route (where demand is zero)
    heuristic_values = potential_improvement * (1 - (cumulative_demand == demands).float())
    
    # Set the diagonal of the heuristic matrix to a large negative value to avoid selecting the depot
    # as the next customer in the route
    diag_mask = torch.eye(distance_matrix.shape[0], dtype=torch.float32)
    heuristic_values += diag_mask * -float('inf')
    
    return heuristic_values