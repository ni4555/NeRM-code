import random
import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Constants for the heuristic
    MAX_DISTANCE = float('inf')
    DEMAND_THRESHOLD = 0.1  # Threshold for considering demand relaxation
    
    # Initialize the heuristics matrix with MAX_DISTANCE
    heuristics = torch.full_like(distance_matrix, fill_value=MAX_DISTANCE)
    
    # Step 1: Apply dynamic window technique by considering only edges to nodes with high demand
    high_demand_nodes = (demands > DEMAND_THRESHOLD).nonzero(as_tuple=True)[0]
    for node in high_demand_nodes:
        heuristics[node, :] = distance_matrix[node, :]
        heuristics[:, node] = distance_matrix[:, node]
    
    # Step 2: Use node partitioning to group customers based on proximity
    # This step requires a more complex partitioning algorithm and is not implemented here
    # For the sake of example, we will randomly assign customers to partitions
    num_partitions = 3
    partitions = torch.randint(0, num_partitions, demands.shape)
    
    # Step 3: Apply multi-objective evolutionary algorithm principles to optimize edge selection
    # We simulate this by randomly assigning weights to each partition
    partition_weights = torch.rand(num_partitions)
    weighted_distances = distance_matrix * partition_weights[partitions]
    
    # Update heuristics matrix with weighted distances
    heuristics = torch.min(heuristics, weighted_distances)
    
    # Step 4: Apply demand relaxation for edges with high potential to reduce distance
    # We simulate this by adding a small positive value to edges that connect high demand nodes
    for node in high_demand_nodes:
        heuristics[node, :] = torch.clamp(heuristics[node, :], min=0)
        heuristics[:, node] = torch.clamp(heuristics[:, node], min=0)
    
    # Step 5: Apply path decomposition to consider only necessary edges
    # This step requires a more complex path decomposition algorithm and is not implemented here
    # For the sake of example, we will randomly remove edges from the heuristics matrix
    num_edges_to_remove = int(0.1 * heuristics.numel())
    edges_to_remove = torch.rand(num_edges_to_remove)
    edges_to_remove = edges_to_remove * heuristics.numel()
    heuristics.view(-1)[edges_to_remove] = MAX_DISTANCE
    
    return heuristics