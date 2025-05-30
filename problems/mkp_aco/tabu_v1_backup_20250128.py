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
        self.n, self.m = weight.shape
        self.prize = prize
        self.weight = weight
        self.n_solutions = n_solutions
        self.tabu_tenure = tabu_tenure
        self.device = device
        
        self.tabu_list = []
        self.best_solution = None
        self.best_obj = 0
        
    def is_feasible(self, solution: torch.Tensor) -> bool:
        """Check if a solution satisfies all constraints"""
        weights = (solution.unsqueeze(1) * self.weight).sum(dim=0)  # [m,]
        return (weights <= 1).all()
    
    def objective(self, solution: torch.Tensor) -> float:
        """Calculate objective value for a solution"""
        return (solution * self.prize).sum()
    
    def generate_initial_solution(self) -> torch.Tensor:
        """Generate a random feasible solution"""
        solution = torch.zeros(self.n, device=self.device)
        indices = torch.randperm(self.n)
        
        for idx in indices:
            temp_sol = solution.clone()
            temp_sol[idx] = 1
            if self.is_feasible(temp_sol):
                solution = temp_sol
                
        return solution
    
    def get_neighborhood(self, solution: torch.Tensor) -> List[Tuple[torch.Tensor, int]]:
        """Generate neighborhood by flipping bits"""
        neighbors = []
        for i in range(self.n):
            new_sol = solution.clone()
            new_sol[i] = 1 - new_sol[i]  # flip bit
            if self.is_feasible(new_sol):
                neighbors.append((new_sol, i))
        return neighbors
    
    def is_tabu(self, move_idx: int) -> bool:
        """Check if a move is in tabu list"""
        return move_idx in self.tabu_list
    
    def update_tabu_list(self, move_idx: int):
        """Add move to tabu list and remove old moves"""
        self.tabu_list.append(move_idx)
        if len(self.tabu_list) > self.tabu_tenure:
            self.tabu_list.pop(0)
    
    @torch.no_grad()
    def run(self, n_iterations: int) -> Tuple[float, torch.Tensor]:
        """Run tabu search for given iterations"""
        current_solution = self.generate_initial_solution()
        self.best_solution = current_solution.clone()
        self.best_obj = self.objective(current_solution)
        
        for _ in range(n_iterations):
            neighbors = self.get_neighborhood(current_solution)
            if not neighbors:
                continue
                
            # Evaluate all neighbors
            best_neighbor = None
            best_neighbor_obj = float('-inf')
            best_move_idx = None
            
            for neighbor, move_idx in neighbors:
                neighbor_obj = self.objective(neighbor)
                
                # Accept if better than best known and not tabu
                # or if tabu but satisfies aspiration criterion
                if (not self.is_tabu(move_idx) and neighbor_obj > best_neighbor_obj) or \
                   (neighbor_obj > self.best_obj):
                    best_neighbor = neighbor
                    best_neighbor_obj = neighbor_obj
                    best_move_idx = move_idx
            
            if best_neighbor is None:
                continue
                
            # Update current solution
            current_solution = best_neighbor
            
            # Update best solution if improved
            if best_neighbor_obj > self.best_obj:
                self.best_solution = best_neighbor.clone()
                self.best_obj = best_neighbor_obj
            
            # Update tabu list
            self.update_tabu_list(best_move_idx)
            
        return self.best_obj, self.best_solution
