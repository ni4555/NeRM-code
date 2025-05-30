import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total distance from the depot to all other nodes
    total_distance_from_depot = distance_matrix[0, 1:]
    
    # Calculate the total distance from all nodes back to the depot
    total_distance_to_depot = distance_matrix[:, 1:].sum(dim=1)
    
    # Calculate the sum of demands for all nodes
    total_demand = demands[1:]
    
    # Calculate the priority score for each edge based on the heuristic
    # We use a simple heuristic where we prioritize edges that have lower total distance
    # and lower demand. We subtract the demand from the score to make it negative
    # for undesirable edges.
    priority_score = -total_distance_from_depot - total_demand
    
    # The depot to itself edge should not be included in the solution, set its score to a large negative value
    priority_score[0] = float('-inf')
    
    return priority_score