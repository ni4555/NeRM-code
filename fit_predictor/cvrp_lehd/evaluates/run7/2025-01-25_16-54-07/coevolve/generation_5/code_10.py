import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total distance for each edge
    total_distance = torch.sum(distance_matrix, dim=1)
    
    # Calculate the return to the depot distance
    return_to_depot = distance_matrix[:, 0]
    
    # Normalize the return to the depot distance by the demand
    normalized_return_to_depot = return_to_depot / demands
    
    # Combine the total distance and normalized return to depot
    combined_score = total_distance + normalized_return_to_depot
    
    # Subtract the demand to encourage selecting edges with lower demand
    combined_score -= demands
    
    # Return the heuristics matrix
    return combined_score