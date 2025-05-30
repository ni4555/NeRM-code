import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demand_normalized = demands / total_capacity
    
    # Calculate normalized distance matrix
    distance_normalized = distance_matrix / distance_matrix.mean()
    
    # Initialize heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Greedy assignment of customers based on normalized demand
    for i in range(1, n):
        min_heap = []
        for j in range(n):
            if i != j:
                # Push edge with initial heuristic value (normalized distance)
                torch.heappush(min_heap, (-distance_normalized[i, j].item(), (i, j)))
        
        # Assign the customer to the vehicle with the lowest load
        while min_heap and demands[torch.tensor(min_heap[0][1][0])] <= total_capacity:
            _, edge = torch.heappop(min_heap)
            heuristic_matrix[edge] = demand_normalized[edge[0]] * demand_normalized[edge[1]]
            total_capacity -= demands[edge[0]]
    
    return heuristic_matrix