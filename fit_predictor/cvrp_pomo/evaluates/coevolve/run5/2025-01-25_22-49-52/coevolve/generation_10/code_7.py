import random
import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Step 1: Demand Relaxation
    # Normalize the demands by the total vehicle capacity (assumed to be 1 for simplicity)
    relaxed_demands = demands / demands.sum()
    
    # Step 2: Node Partitioning
    # Use a simple threshold to partition nodes based on their demands
    threshold = 0.1  # Threshold can be adjusted for different scenarios
    partitioned_nodes = relaxed_demands > threshold
    
    # Step 3: Dynamic Window Approach
    # Create a dynamic window that will be updated dynamically
    dynamic_window = torch.zeros(n)
    
    # Step 4: Multi-Objective Evolutionary Algorithm (MOEA) Indicators
    # For simplicity, we will use a random heuristic to determine the promising edges
    # In practice, this should be replaced by a proper MOEA implementation
    moea_indicators = torch.rand(n)
    
    # Step 5: Constraint Programming Heuristic
    # For each node, calculate the heuristic based on distance, demand, and other factors
    heuristic_values = distance_matrix * (1 - relaxed_demands) + distance_matrix * partitioned_nodes * 2
    heuristic_values = heuristic_values * moea_indicators
    
    # Step 6: Apply the Dynamic Window to Adjust Heuristic Values
    # Update the dynamic window based on some rule (e.g., recent changes in demands)
    # For simplicity, we'll just add the current heuristic values to the window
    dynamic_window = dynamic_window + heuristic_values
    
    # Step 7: Apply the Demand Relaxation to the Heuristic Values
    # Adjust the heuristic values based on the relaxed demands
    adjusted_heuristic_values = heuristic_values * relaxed_demands
    
    # Step 8: Final Heuristic Calculation
    # Combine all the heuristics to get the final heuristic values
    final_heuristic_values = adjusted_heuristic_values + dynamic_window
    
    # Step 9: Normalize the Final Heuristic Values
    # Ensure that the heuristic values are within the desired range
    final_heuristic_values = final_heuristic_values - final_heuristic_values.max()
    final_heuristic_values = final_heuristic_values / final_heuristic_values.abs().max()
    
    return final_heuristic_values