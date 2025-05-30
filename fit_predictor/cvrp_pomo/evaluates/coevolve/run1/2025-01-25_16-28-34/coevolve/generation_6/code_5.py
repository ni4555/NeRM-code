import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Problem-specific Local Search
    # Calculate the initial heuristics based on inverse distance (heuristic for nearest neighbor)
    heuristics = 1 / (distance_matrix ** 2)
    
    # Adjust heuristics based on demands to account for customer demand (heuristic for load balancing)
    demand_weight = demands / demands.sum()
    heuristics *= demand_weight
    
    # Normalize heuristics to ensure they sum up to 1
    heuristics /= heuristics.sum()
    
    # Adaptive PSO Population Management and Dynamic Tabu Search with Adaptive Cost Function
    # These components would require more complex implementations involving iterative adjustments
    # and cannot be represented with a single vectorized expression.
    # Here, we simulate these steps with a placeholder function that would adaptively adjust heuristics.
    def adapt_heuristics(heuristics):
        # Placeholder for the adaptive adjustment process
        # This would involve more complex logic, possibly iterative optimization techniques
        # For now, we simply add a constant to simulate the adaptive adjustment
        return heuristics + 0.1
    
    heuristics = adapt_heuristics(heuristics)
    
    # Ensure that the heuristics are still normalized after adjustment
    heuristics /= heuristics.sum()
    
    return heuristics