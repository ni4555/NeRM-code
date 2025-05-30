import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Assuming demand is already normalized to be within the vehicle capacity.
    vehicle_capacity = demands.sum()
    
    # Initialize a heuristics matrix with zeros.
    heuristics_matrix = torch.zeros((n, n), dtype=torch.float32)
    
    # Calculate the score for each edge.
    for i in range(n):
        for j in range(n):
            # If i is the depot (0), ignore the first row
            if i == 0:
                continue
            
            # Calculate the current edge score
            score = 0
            
            # Check if there's a demand for this customer and add a score if true
            if demands[j] > 0:
                score += 1  # Assuming 1 is a positive score for promising edges
            
            # Apply a penalty for longer distances
            # This is an arbitrary choice of 10 as the penalty, but it can be tuned
            if j > 0:  # Avoiding the depot node (0) as it has no distance
                score -= distance_matrix[i, j] * 10  # 10 is the penalty factor for distance
            
            # Store the score in the heuristics matrix
            heuristics_matrix[i, j] = score
            
    return heuristics_matrix