import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Constraint Programming Heuristic: Calculate the relative importance of each edge
    # based on the ratio of the demand of the destination node to the sum of demands
    # from the source node to the destination node.
    cp_heuristic = demands[1:] / demands[:-1]
    
    # Dynamic Window Approach: Normalize the heuristic values based on the current
    # vehicle capacities to account for dynamic changes in demand and vehicle capacity.
    cp_heuristic = cp_heuristic / demands.sum()
    
    # Multi-Objective Evolutionary Algorithm: Introduce a penalty for high demand nodes
    # to ensure a balance between minimizing route distances and maintaining vehicle loads.
    # This could be done by scaling the heuristic values inversely proportional to the
    # normalized demand of the nodes.
    cp_heuristic = cp_heuristic * (1 / demands[1:])
    
    # Demand Relaxation: Relax the demand of the destination node for each edge
    # to handle dynamic changes in node demands, by subtracting a small fraction of the
    # demand.
    cp_heuristic = cp_heuristic - (demands[1:] / demands.sum())
    
    # Ensure that the heuristic values are non-negative and not too high to be
    # undesirable, while still being informative.
    cp_heuristic = torch.clamp(cp_heuristic, min=-1.0, max=1.0)
    
    # Node Partitioning: Since the depot is indexed by 0, we can assume that the first
    # row and column of the distance matrix are the edges to and from the depot.
    # We can adjust the heuristic values for these edges to prioritize them.
    cp_heuristic[0, :] = cp_heuristic[0, :] + 1.5
    cp_heuristic[:, 0] = cp_heuristic[:, 0] + 1.5
    
    # Return the final heuristic values as a tensor of the same shape as the distance matrix.
    return cp_heuristic