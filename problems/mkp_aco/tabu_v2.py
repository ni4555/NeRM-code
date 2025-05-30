import torch
import numpy as np
from typing import Tuple, List

class TabuSearch:
    def __init__(self,
                 prize,      # shape [n,]
                 weight,     # shape [m, n]
                 heuristic,  # not used in tabu search
                 n_solutions=30,  # number of initial solutions
                 tabu_tenure=7,   # length of tabu list
                 device='cpu'
                 ):
        # Ensure prize is on the correct device and has the right shape
        self.prize = torch.as_tensor(prize, device=device)
        if len(self.prize.shape) == 0:
            self.prize = self.prize.unsqueeze(0)
        
        # Ensure weight is on the correct device and has correct shape
        self.weight = torch.as_tensor(weight, device=device)
        # If weight shape doesn't match prize, try transposing it
        if self.weight.shape[1] != self.prize.shape[0]:
            self.weight = self.weight.T
        
        self.m, self.n = self.weight.shape
        
        # Verify dimensions match
        if self.prize.shape[0] != self.n:
            raise ValueError(f"Prize shape {self.prize.shape[0]} must match weight shape[1] {self.n}")
        
        self.n_solutions = n_solutions
        self.tabu_tenure = tabu_tenure
        self.device = device
        
        self.tabu_list = []
        self.best_solution = None
        self.best_obj = 0
        
    def is_feasible(self, solution: torch.Tensor) -> bool:
        """Check if a solution satisfies all constraints"""
        # weight shape is [m, n], solution shape is [n]
        # Need to multiply and sum along n dimension
        weights = (solution * self.weight).sum(dim=1)  # [m,]
        return (weights <= 1).all()
    
    def objective(self, solution: torch.Tensor) -> float:
        """Calculate objective value for a solution"""
        return (solution * self.prize).sum()
    
    def generate_initial_solution(self) -> torch.Tensor:
        """Generate initial solution using greedy approach"""
        solution = torch.zeros(self.n, device=self.device)
        
        # Calculate average efficiency (prize/average_weight ratio) for each item
        # weight shape is [m, n], we want to get average weight per item
        avg_weights = self.weight.mean(dim=0)  # [n,]
        efficiency = torch.zeros_like(self.prize, device=self.device)
        nonzero_mask = avg_weights > 0
        efficiency[nonzero_mask] = self.prize[nonzero_mask] / avg_weights[nonzero_mask]
        
        # Sort items by efficiency
        sorted_indices = torch.argsort(efficiency, descending=True)
        
        for idx in sorted_indices:
            temp_sol = solution.clone()
            temp_sol[idx] = 1
            if self.is_feasible(temp_sol):
                solution = temp_sol
                
        return solution
    
    def get_neighborhood(self, solution: torch.Tensor) -> List[Tuple[torch.Tensor, List[int]]]:
        """Generate neighborhood using both 1-flip and 2-flip moves"""
        neighbors = []
        
        # 1-flip neighbors
        for i in range(self.n):
            new_sol = solution.clone()
            new_sol[i] = 1 - new_sol[i]
            if self.is_feasible(new_sol):
                neighbors.append((new_sol, [i]))
        
        # 2-flip neighbors (randomly sample pairs to avoid too many neighbors)
        n_pairs = min(self.n * 2, 100)  # Limit number of pairs to check
        pairs = torch.randint(0, self.n, (n_pairs, 2))
        for i, j in pairs:
            if i != j:
                new_sol = solution.clone()
                new_sol[i] = 1 - new_sol[i]
                new_sol[j] = 1 - new_sol[j]
                if self.is_feasible(new_sol):
                    neighbors.append((new_sol, [i.item(), j.item()]))
        
        return neighbors
    
    def is_tabu(self, move_indices: List[int]) -> bool:
        """Check if any move in the combination is tabu"""
        return any(idx in self.tabu_list for idx in move_indices)
    
    def update_tabu_list(self, move_indices: List[int]):
        """Add moves to tabu list and remove old moves"""
        for idx in move_indices:
            self.tabu_list.append(idx)
        while len(self.tabu_list) > self.tabu_tenure:
            self.tabu_list.pop(0)
    
    @torch.no_grad()
    def run(self, n_iterations: int) -> Tuple[float, torch.Tensor]:
        """Run tabu search with diversification"""
        # Generate multiple initial solutions
        solutions = [self.generate_initial_solution() for _ in range(self.n_solutions)]
        current_solution = max(solutions, key=self.objective)
        self.best_solution = current_solution.clone()
        self.best_obj = self.objective(current_solution)
        
        # Counter for diversification
        stagnation_counter = 0
        last_improvement = 0
        
        for iter_idx in range(n_iterations):
            neighbors = self.get_neighborhood(current_solution)
            if not neighbors:
                continue
                
            # Evaluate all neighbors
            best_neighbor = None
            best_neighbor_obj = float('-inf')
            best_move_indices = None
            
            for neighbor, move_indices in neighbors:
                neighbor_obj = self.objective(neighbor)
                
                # Accept if better than best known and not tabu
                # or if tabu but satisfies aspiration criterion
                if (not self.is_tabu(move_indices) and neighbor_obj > best_neighbor_obj) or \
                   (neighbor_obj > self.best_obj):
                    best_neighbor = neighbor
                    best_neighbor_obj = neighbor_obj
                    best_move_indices = move_indices
            
            if best_neighbor is None:
                continue
                
            # Update current solution
            current_solution = best_neighbor
            
            # Update best solution if improved
            if best_neighbor_obj > self.best_obj:
                self.best_solution = best_neighbor.clone()
                self.best_obj = best_neighbor_obj
                last_improvement = iter_idx
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Diversification strategy
            if stagnation_counter > 20:  # If no improvement for 20 iterations
                # Generate new solution with some probability
                if torch.rand(1).item() < 0.3:
                    current_solution = self.generate_initial_solution()
                    self.tabu_list.clear()  # Clear tabu list for fresh start
                stagnation_counter = 0
            
            # Update tabu list
            self.update_tabu_list(best_move_indices)
            
        return self.best_obj, self.best_solution
