import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Get the size of the distance matrix
    n = distance_matrix.shape[0]
    
    # Initialize a tensor to store the heuristic values, with the same shape as the distance matrix
    heuristic_values = torch.zeros_like(distance_matrix)
    
    # Node partitioning
    # For simplicity, we'll use a threshold-based approach for demonstration purposes.
    # This could be replaced by a more sophisticated partitioning algorithm.
    threshold = torch.mean(demands)  # A simple threshold based on the average demand
    partitioned_nodes = torch.where(demands > threshold, 1, 0)
    
    # Demand relaxation
    # We relax the demands slightly to make the problem more tractable.
    relaxed_demands = demands * 0.95
    
    # Initialize a variable to store the cumulative demand for each node
    cumulative_demand = torch.zeros(n)
    
    # Calculate the cumulative demand for each node considering relaxed demands
    for i in range(n):
        cumulative_demand[i] = relaxed_demands[partitioned_nodes == i].sum()
    
    # Dynamic window approach (simulated here as a sliding window of the heuristic calculation)
    window_size = 3  # Size of the window for dynamic calculation
    for i in range(n):
        for j in range(n):
            # Calculate the heuristic value for edge (i, j)
            if i != j:  # Avoid the depot node
                edge_demand = relaxed_demands[j]
                if cumulative_demand[i] + edge_demand <= demands[i] and cumulative_demand[j] + demands[i] <= demands[j]:
                    # If the vehicle can carry the extra demand, add the distance to the heuristic value
                    heuristic_value = distance_matrix[i, j] - (1 - partitioned_nodes[i] * partitioned_nodes[j])
                else:
                    # If not, mark this edge as undesirable with a large negative value
                    heuristic_value = -float('inf')
            else:
                # If it's the same node, there's no need for an edge, so set the heuristic to a large negative value
                heuristic_value = -float('inf')
            
            # Apply the dynamic window approach
            window = distance_matrix[i, max(0, j-window_size):j+1]
            heuristic_value += window.mean()
            
            # Update the heuristic value for edge (i, j)
            heuristic_values[i, j] = heuristic_value
    
    return heuristic_values