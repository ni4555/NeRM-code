import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Assuming demand is normalized and the depot is node 0
    demand_vector = torch.cat((torch.tensor(0.0), demands))
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristics using the Euclidean TSP heuristic approach
    # Each edge heuristics value is calculated as the negative of the distance
    # minus the negative of the demand difference
    for i in range(n):
        for j in range(n):
            if i != j:
                # For depot node 0, the demand is 0, hence the demand difference is 0
                heuristics[i][j] = -distance_matrix[i][j] - (demand_vector[j] - demand_vector[i])
            else:
                heuristics[i][j] = float('-inf')
    
    return heuristics