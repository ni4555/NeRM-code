import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Assuming that the depot is at index 0 and is not served by a vehicle
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Demand Relaxation: Normalize demands to fit vehicle capacity
    vehicle_capacity = 1.0  # For simplicity, assume vehicle capacity is the total demand
    demands = demands / vehicle_capacity
    
    # Node Partitioning: A simple k-means-like approach (simplified)
    # For demonstration, we'll just create two clusters for simplicity
    # In a real-world scenario, you would use a more sophisticated clustering method
    centroids = torch.tensor([0.1, 0.9])  # Randomly chosen centroids for two clusters
    cluster_distances = torch.cdist(centroids, distance_matrix)
    cluster_assignment = torch.argmin(cluster_distances, dim=1)
    
    # Assign heuristic values based on cluster assignment
    for i in range(1, n):  # Skip the depot node
        if cluster_assignment[i] == 0:
            # Node is in the first cluster
            heuristic_matrix[i, :i] = -distance_matrix[i, :i]
            heuristic_matrix[:i, i] = -distance_matrix[:i, i]
        else:
            # Node is in the second cluster
            heuristic_matrix[i, :i] = distance_matrix[i, :i]
            heuristic_matrix[:i, i] = distance_matrix[:i, i]
    
    return heuristic_matrix

# Example usage:
distance_matrix = torch.tensor([[0, 2, 5, 3],
                                [2, 0, 1, 4],
                                [5, 1, 0, 2],
                                [3, 4, 2, 0]], dtype=torch.float32)
demands = torch.tensor([0.2, 0.1, 0.3, 0.4], dtype=torch.float32)

# Call the function
heuristic_matrix = heuristics_v2(distance_matrix, demands)
print(heuristic_matrix)