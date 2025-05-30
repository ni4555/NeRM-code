import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total distance from the depot to all other nodes
    total_distance_from_depot = distance_matrix[0, 1:]
    
    # Calculate the total distance from all nodes back to the depot
    total_distance_to_depot = distance_matrix[:, 1:].sum(dim=1)
    
    # Calculate the total demand from all nodes
    total_demand = demands[1:]
    
    # Normalize the demand to account for vehicle capacity
    normalized_demand = (demands[1:] / total_demand.sum()).unsqueeze(1)
    
    # Calculate the potential idle time for each node
    potential_idle_time = total_distance_to_depot * normalized_demand
    
    # The heuristic score is the negative of the potential idle time
    # since we want to minimize idle time (hence the negative sign)
    heuristic_scores = -potential_idle_time
    
    # Add the depot's own distance to itself (which is 0) to the scores
    heuristic_scores = torch.cat([torch.zeros(1), heuristic_scores])
    
    return heuristic_scores