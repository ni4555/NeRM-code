import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    depot_index = 0  # The depot is at index 0
    
    # Step 1: Initialize a basic heuristic - using inverse distance to the depot
    # This heuristic assumes that closer customers are more desirable to visit
    heuristics = -torch.abs(distance_matrix[depot_index, :])  # Negative distance for vectorization purposes

    # Step 2: Adjust the heuristic based on demands
    # Promote closer nodes with lower demands as more desirable
    heuristics += (1 / demands)
    
    return heuristics