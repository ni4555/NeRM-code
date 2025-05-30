import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix to have a scale between 0 and 1
    max_distance = distance_matrix.max()
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Calculate the sum of demands in each row (except the first row, which is the depot)
    demand_sum = demands.cumsum(0)[1:]
    
    # Calculate the load on each route (excluding the depot)
    load = torch.clamp(normalized_distance_matrix * demands[1:], min=0)
    load_sum = load.cumsum(0)[1:]
    
    # Calculate the heuristic value for each edge
    # The heuristic is designed to reward short distances and negative load balance
    heuristic_matrix = torch.where(
        load_sum < demands[1:], 
        -load_sum - distance_matrix,  # Negative load on the edge is penalized
        distance_matrix  # Positive load balance is rewarded
    )
    
    return heuristic_matrix