from typing import Optional
import logging
import subprocess
import numpy as np
import os
from omegaconf import DictConfig
from pathlib import Path
import time

from utils.utils import *
from utils.llm_client.base import BaseClient

class NoRevolutionGenerator:
    def __init__(
        self, 
        cfg: DictConfig, 
        root_dir: str, 
        generator_llm: BaseClient,
    ) -> None:
        # Initialize timing attributes
        self.api_log_path = Path("api_invoke_log.log")
        self.eval_batch_log_path = Path("eval_time_log_batch.log")
        self.eval_single_log_path = Path("eval_time_log_single.log")
        self.obj_time_log_path = Path("obj_time.log")
        
        # Initialize timing lists
        self.api_call_times = []
        self.eval_batch_times = []
        self.eval_single_times = []
        self.start_time = time.time()
        
        # Load existing timing data
        self._load_existing_timing_data()
        
        # Basic initialization
        self.cfg = cfg
        self.generator_llm = generator_llm
        self.root_dir = root_dir
        
        self.function_evals = 0
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None
        self.best_individuals_overall = []
        
        self.init_prompt()

    def _load_existing_timing_data(self):
        """Load existing timing data from log files if they exist"""
        for log_path, times_list in [
            (self.api_log_path, self.api_call_times),
            (self.eval_batch_log_path, self.eval_batch_times),
            (self.eval_single_log_path, self.eval_single_times)
        ]:
            if log_path.exists():
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[1:]
                    times_list.extend([float(line.strip()) for line in lines])

    def _log_timing(self, duration: float, log_type: str):
        """Log timing information"""
        if log_type == "api":
            self.api_call_times.append(duration)
            log_path = self.api_log_path
            times_list = self.api_call_times
        elif log_type == "eval_batch":
            self.eval_batch_times.append(duration)
            log_path = self.eval_batch_log_path
            times_list = self.eval_batch_times
        else:  # eval_single
            self.eval_single_times.append(duration)
            log_path = self.eval_single_log_path
            times_list = self.eval_single_times
            
        total_time = sum(times_list)
        total_count = len(times_list)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Total count: {total_count}, Total time: {total_time:.2f}s\n")
            for t in times_list:
                f.write(f"{t:.4f}\n")

    def _update_best_individuals(self, individual: dict) -> None:
        """Update best individuals list and log objective values"""
        if not individual.get("exec_success", False):
            return
        
        self.best_individuals_overall.append(individual)
        self.best_individuals_overall.sort(key=lambda x: x.get("obj", float("inf")))
        self.best_individuals_overall = self.best_individuals_overall[:5]
        
        if self.best_obj_overall is None or individual["obj"] < self.best_obj_overall:
            self.best_obj_overall = individual["obj"]
            self.best_code_overall = individual["code"]
            self.best_code_path_overall = individual["code_path"]
        
        self._log_obj_time()

    def _log_obj_time(self):
        """Log objective values and timing information"""
        current_time = time.time() - self.start_time
        
        top_5_avg = float('inf')
        if len(self.best_individuals_overall) >= 5:
            top_5_avg = sum(ind["obj"] for ind in self.best_individuals_overall[:5]) / 5
        
        current_best = self.best_obj_overall if self.best_obj_overall is not None else float('inf')
        
        with open(self.obj_time_log_path, 'a') as f:
            f.write(f"{current_time:.2f}\t{current_best:.6f}\t{top_5_avg:.6f}\n")

    def init_prompt(self) -> None:
        """Initialize prompts and problem-specific information"""
        self.problem = self.cfg.problem.problem_name
        self.problem_desc = self.cfg.problem.description
        self.problem_size = self.cfg.problem.problem_size
        self.func_name = self.cfg.problem.func_name
        self.obj_type = self.cfg.problem.obj_type
        self.problem_type = self.cfg.problem.problem_type
        
        logging.info(f"Problem: {self.problem}")
        logging.info(f"Problem description: {self.problem_desc}")
        logging.info(f"Function name: {self.func_name}")
        
        self.prompt_dir = f"{self.root_dir}/prompts"
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt.py"
        
        # Load problem-specific prompts
        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        problem_prompt_path = f'{self.prompt_dir}/{self.problem}{prompt_path_suffix}'
        self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt')
        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')
        
        # Load common prompts
        self.system_generator_prompt = file_to_string(f'{self.prompt_dir}/common/system_generator.txt')
        self.user_generator_prompt = file_to_string(f'{self.prompt_dir}/common/user_generator.txt').format(
            func_name=self.func_name, 
            problem_desc=self.problem_desc,
            func_desc=self.func_desc,
        )

    def evolve(self):
        """Main loop for generating and evaluating solutions"""
        iteration = 0
        batch_size = self.cfg.pop_size  # Use population size as batch size
        
        while self.function_evals < self.cfg.max_fe:
            # Generate multiple solutions in parallel
            start_time = time.time()
            system = self.system_generator_prompt
            user = self.user_generator_prompt
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            
            # Generate batch_size responses in parallel
            responses = self.generator_llm.multi_chat_completion([messages], batch_size)
            api_duration = time.time() - start_time
            self._log_timing(api_duration, "api")
            
            # Convert responses to individuals
            individuals = []
            for response_id, response in enumerate(responses):
                individual = {
                    "stdout_filepath": f"problem_iter{iteration}_stdout{response_id}.txt",
                    "code_path": f"problem_iter{iteration}_code{response_id}.py",
                    "code": extract_code_from_generator(response),
                    "response_id": response_id,
                }
                individuals.append(individual)
            
            # Evaluate batch of individuals in parallel
            evaluated_individuals = self.evaluate_batch(individuals)
            self.function_evals += len(evaluated_individuals)
            
            # Update best individuals
            for individual in evaluated_individuals:
                if individual["exec_success"]:
                    self._update_best_individuals(individual)
            
            # Log progress
            logging.info(f"Iteration {iteration}, Evaluations {self.function_evals}: Best obj so far: {self.best_obj_overall}")
            
            iteration += 1
            
            # Check if target objective value is reached
            if hasattr(self.cfg, 'exp_obj_test_only'):
                if self.best_obj_overall is not None and self.best_obj_overall <= self.cfg.exp_obj_test_only:
                    logging.info(f"Reached target objective value: {self.best_obj_overall}")
                    break
        
        return self.best_code_overall, self.best_code_path_overall

    def evaluate_batch(self, individuals: list[dict]) -> list[dict]:
        """Evaluate a batch of individuals in parallel"""
        batch_start = time.time()
        
        # Start all processes
        processes = []
        for individual in individuals:
            try:
                process = self._run_code(individual)
                processes.append((process, individual, time.time()))
            except Exception as e:
                individual = self.mark_invalid_individual(individual, str(e))
                processes.append((None, individual, None))
        
        # Wait for all processes to complete
        for process, individual, start_time in processes:
            if process is None:
                continue
            
            try:
                process.communicate(timeout=self.cfg.timeout)
                single_duration = time.time() - start_time
                self._log_timing(single_duration, "eval_single")
                
                with open(individual["stdout_filepath"], 'r') as f:
                    stdout_str = f.read()
                traceback_msg = filter_traceback(stdout_str)
                
                if traceback_msg == '':
                    try:
                        individual["obj"] = float(stdout_str.split('\n')[-2]) if self.obj_type == "min" else -float(stdout_str.split('\n')[-2])
                        individual["exec_success"] = True
                    except:
                        individual = self.mark_invalid_individual(individual, "Invalid stdout / objective value!")
                else:
                    individual = self.mark_invalid_individual(individual, traceback_msg)
                    
            except subprocess.TimeoutExpired as e:
                individual = self.mark_invalid_individual(individual, str(e))
                process.kill()
        
        batch_duration = time.time() - batch_start
        self._log_timing(batch_duration, "eval_batch")
        
        return [ind for _, ind, _ in processes]

    def _run_code(self, individual: dict) -> subprocess.Popen:
        """Run the generated code"""
        with open(self.output_file, 'w') as file:
            file.writelines(individual["code"] + '\n')

        eval_file_path = f'{self.root_dir}/problems/{self.problem}/eval.py'
        if self.problem_type == "black_box":
            eval_file_path = f'{self.root_dir}/problems/{self.problem}/eval_black_box.py'
            
        with open(individual["stdout_filepath"], 'w') as f:
            process = subprocess.Popen(
                ['python', '-u', eval_file_path, f'{self.problem_size}', self.root_dir, "train"],
                stdout=f, 
                stderr=f
            )

        block_until_running(individual["stdout_filepath"], log_status=True)
        return process

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """Mark an individual as invalid"""
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual
