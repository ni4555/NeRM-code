import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    import torch
    import numpy as np

    n = distance_matrix.shape[0]
    total_demand = demands.sum().item()
    heuristics = torch.zeros_like(distance_matrix)

    # Create a mask for nodes with positive demand
    demand_mask = (demands > 0).float()

    # Iterate over all pairs of nodes
    for i in range(1, n):
        for j in range(1, n):
            if demand_mask[i] > 0 and demand_mask[j] > 0:
                # Calculate the potential heuristics value for this edge
                edge_heuristic = demands[j] - demands[i]
                # Scale the heuristic by the inverse of the demand for normalization
                edge_heuristic /= demands[i]
                # Adjust the heuristic to account for distance
                edge_heuristic *= distance_matrix[i][j]
                # Apply the mask to keep only positive heuristics
                heuristics[i][j] = torch.clamp(edge_heuristic, min=0)
                heuristics[j][i] = heuristics[i][j]

    return heuristics
