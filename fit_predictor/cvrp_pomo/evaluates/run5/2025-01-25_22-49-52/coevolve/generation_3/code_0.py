import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Step 1: Node partitioning
    # Partition nodes based on a simple heuristic, e.g., using k-means clustering on demand
    k = 5  # Number of clusters can be tuned
    _, partition_indices = torch.unique(torch.argmax(demands, dim=0)[:k], return_inverse=True)
    
    # Step 2: Demand relaxation
    # Relax the demands to improve the initial solution quality
    relaxed_demands = demands / partition_indices.float() * k
    
    # Step 3: Path decomposition
    # Decompose the problem into smaller subproblems, e.g., by considering only edges within each cluster
    within_cluster_distances = torch.zeros_like(distance_matrix)
    for i in range(k):
        within_cluster_distances[partition_indices == i] = distance_matrix[partition_indices == i]
    
    # Step 4: Multi-objective evolutionary algorithm
    # Generate a set of promising paths using a multi-objective evolutionary approach
    # For simplicity, we will use a weighted sum approach to combine distance and demand
    # Here we simulate this step by simply using a weighted sum of distances and relaxed demands
    weight_distance = 0.6
    weight_demand = 0.4
    path_scores = (within_cluster_distances * weight_distance + relaxed_demands * weight_demand).sum(dim=1)
    
    # Step 5: Dynamic window approach
    # Adjust the heuristic based on the current vehicle capacities
    # Here we simulate this step by adjusting the heuristic scores based on a simple threshold
    capacity_threshold = 0.8  # Vehicle capacity threshold
    path_scores *= torch.clamp((demands / relaxed_demands), min=1.0, max=capacity_threshold)
    
    # Step 6: Calculate heuristic values for each edge
    heuristic_values = torch.zeros_like(distance_matrix)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j and partition_indices[i] == partition_indices[j]:
                # If the nodes are in the same cluster and the edge is within the cluster
                heuristic_values[i, j] = path_scores[i]
            else:
                # If the edge is not within the same cluster, it's less promising
                heuristic_values[i, j] = -1
    
    return heuristic_values