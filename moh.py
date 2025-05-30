# -*- coding: utf-8 -*-
from typing import Optional, List, Tuple, Dict
import logging
import subprocess
import numpy as np
import os
import json
import yaml
from omegaconf import DictConfig
from pathlib import Path
import time
from datetime import datetime
import random
import signal

from utils.utils import *
from utils.llm_client.base import BaseClient

class MoH:
    def __init__(
        self, 
        cfg: DictConfig, 
        root_dir: str, 
        generator_llm: BaseClient, 
        reflector_llm: Optional[BaseClient] = None,
    ) -> None:
        # Initialize timing attributes first
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
        self.reflector_llm = reflector_llm or generator_llm
        self.root_dir = root_dir
        
        # Problem specific attributes
        self.problem = cfg.problem.problem_name
        self.problem_desc = cfg.problem.description
        self.problem_size = cfg.problem.problem_size
        self.func_name = cfg.problem.func_name
        self.obj_type = cfg.problem.obj_type
        self.problem_type = cfg.problem.problem_type
        
        # Evolution parameters
        self.mutation_rate = cfg.mutation_rate
        self.iteration = 0
        self.function_evals = 0
        self.elitist = None
        self.long_term_reflection_str = ""
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None
        self.best_individuals_overall = []
        
        # Description evolution parameters
        self.num_generations = 10
        self.population_size = 4
        self.eval_batch_size = 4
        self.num_samples = 12
        self.elite_size = 2
        
        # Create output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"moh_output_{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load YAML config
        self.yaml_path = Path(root_dir) / "cfg" / "problem" / f"{self.problem}.yaml"
        with open(self.yaml_path, 'r') as f:
            self.problem_config = yaml.safe_load(f)
            self.initial_desc = self.problem_config['description']
        
        # Initialize performance predictor
        self.performance_predictor = None
        predictor_model_path = f"{root_dir}/predictors/{self.problem}/best_performance_predictor.pth"
        if os.path.exists(predictor_model_path):
            from predictors.code_performance_predictor_simple import CodePerformancePredictor
            self.performance_predictor = CodePerformancePredictor()
            self.performance_predictor.load_model(predictor_model_path)
            logging.info("Loaded code performance predictor")
        else:
            logging.warning("Performance predictor model not found. Will evaluate all generated code.")
        
        # Initialize prompts
        self.init_prompt()
        
        # Initialize embedding cache
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize code history pool
        self.code_history_pool = []
        self.max_history_pool_size = 30
        
        # Diversity parameters
        self.diversity_threshold = 0.92
        self.min_diversity_samples = 4
        self.max_retries = 3
        
        # Base temperature
        self.base_temperature = getattr(generator_llm, 'temperature', 1)
        
        # Add new attributes for tracking unique fitness values
        self.hist_population = []  # Track historical population with valid fitness
        self.min_unique_fitness_count = 12  # Minimum number of unique fitness values needed

    def init_prompt(self) -> None:
        self.problem = self.cfg.problem.problem_name
        self.problem_desc = self.cfg.problem.description
        self.problem_size = self.cfg.problem.problem_size
        self.func_name = self.cfg.problem.func_name
        self.obj_type = self.cfg.problem.obj_type
        self.problem_type = self.cfg.problem.problem_type
        
        logging.info("Problem: " + self.problem)
        logging.info("Problem description: " + self.problem_desc)
        logging.info("Function name: " + self.func_name)
        
        self.prompt_dir = f"{self.root_dir}/prompts"
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt.py"
        
        # Loading all text prompts
        # Problem-specific prompt components
        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        problem_prompt_path = f'{self.prompt_dir}/{self.problem}{prompt_path_suffix}'
        self.seed_func = file_to_string(f'{problem_prompt_path}/seed_func.txt')
        self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt')
        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')
        if os.path.exists(f'{problem_prompt_path}/external_knowledge.txt'):
            self.external_knowledge = file_to_string(f'{problem_prompt_path}/external_knowledge.txt')
            self.long_term_reflection_str = self.external_knowledge
        else:
            self.external_knowledge = ""
        
        
        # Common prompts
        self.system_generator_prompt = file_to_string(f'{self.prompt_dir}/common/system_generator.txt')
        self.system_reflector_prompt = file_to_string(f'{self.prompt_dir}/common/system_reflector.txt')
        self.user_reflector_st_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_st.txt') if self.problem_type != "black_box" else file_to_string(f'{self.prompt_dir}/common/user_reflector_st_black_box.txt') # shrot-term reflection
        self.user_reflector_lt_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_lt.txt') # long-term reflection
        self.crossover_prompt = file_to_string(f'{self.prompt_dir}/common/crossover.txt')
        self.mutataion_prompt = file_to_string(f'{self.prompt_dir}/common/mutation.txt')
        self.user_generator_prompt = file_to_string(f'{self.prompt_dir}/common/user_generator.txt').format(
            func_name=self.func_name, 
            problem_desc=self.problem_desc,
            func_desc=self.func_desc,
            )
        self.seed_prompt = file_to_string(f'{self.prompt_dir}/common/seed.txt').format(
            seed_func=self.seed_func,
            func_name=self.func_name,
        )

        # Flag to print prompts
        self.print_crossover_prompt = True # Print crossover prompt for the first iteration
        self.print_mutate_prompt = True # Print mutate prompt for the first iteration
        self.print_short_term_reflection_prompt = True # Print short-term reflection prompt for the first iteration
        self.print_long_term_reflection_prompt = True # Print long-term reflection prompt for the first iteration

    def response_to_individual(self, response: str, response_id: int, gen_dir: Path) -> dict:
        """Convert response to individual with generation-specific paths"""
        # Write response to file in generation directory
        file_name = gen_dir / f"response_{response_id}.txt"
        with open(file_name, 'w') as file:
            file.writelines(response + '\n')

        code = extract_code_from_generator(response)

        # Create paths relative to generation directory
        stdout_filepath = gen_dir / f"stdout_{response_id}.txt"
        code_path = gen_dir / f"code_{response_id}.py"
        
        individual = {
            "stdout_filepath": str(stdout_filepath),
            "code_path": str(code_path),
            "code": code,
            "response_id": response_id,
        }
        return individual

    def init_population(self, gen_dir: Path) -> None:
        """Initialize population with generation-specific paths"""
        # Evaluate the seed function, and set it as Elite
        logging.info("Evaluating seed function...")
        code = extract_code_from_generator(self.seed_func).replace("v1", "v2")
        logging.info("Seed function code: \n" + code)
        seed_ind = self.response_to_individual(self.seed_func, 0, gen_dir)
        
        self.seed_ind = seed_ind
        self.population = self.evaluate_population([seed_ind], gen_dir)

        # If seed function is invalid, stop
        if not self.seed_ind["exec_success"]:
            raise RuntimeError(f"Seed function is invalid. Please check the stdout file in {gen_dir}")

        self.update_iter()
        
        # Generate responses
        system = self.system_generator_prompt
        user = self.user_generator_prompt + "\n" + self.seed_prompt + "\n" + self.long_term_reflection_str
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        logging.info("Initial Population Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)

        responses = self.generator_llm.multi_chat_completion(
            [messages], 
            self.cfg.init_pop_size, 
            temperature=self.generator_llm.temperature + 0.3
        )
        population = [self.response_to_individual(response, i, gen_dir) 
                     for i, response in enumerate(responses)]

        # Run code and evaluate population
        population = self.evaluate_population(population, gen_dir)

        # Update iteration
        self.population = population
        self.update_iter()

    def run_code_evolution(self, description: str, gen_dir: Path) -> Dict:
        """Run code evolution process for current description"""
        # Initialize ReEvo-style evolution with generation directory
        self.init_population(gen_dir)
        
        # Create evolution subdirectory for this generation
        evo_dir = gen_dir / "evolution"
        evo_dir.mkdir(exist_ok=True)
        
        # Track starting function evaluations for this description
        start_fe = self.function_evals
        iteration = 0
        
        while True:
            # Check if we've reached max_fe
            if self.function_evals - start_fe >= self.cfg.max_fe:
                logging.info(f"Reached maximum function evaluations ({self.cfg.max_fe}) for current description")
                break
                
            # Check if we have enough unique fitness values
            unique_fitness_values = set(ind["obj"] for ind in self.hist_population 
                                     if ind.get("obj", float("inf")) != float("inf"))
            # if len(unique_fitness_values) >= self.min_unique_fitness_count:
            #     logging.info(f"Found {len(unique_fitness_values)} unique fitness values - stopping evolution")
            #     break
            
            # Create iteration directory
            iter_dir = evo_dir / f"iteration_{iteration}"
            iter_dir.mkdir(exist_ok=True)
            
            # Selection
            population_to_select = self.population if (self.elitist is None or self.elitist in self.population) else [self.elitist] + self.population
            selected_population = self.random_select(population_to_select)
            if selected_population is None:
                break
                
            # Evolution steps
            crossed_population = self.crossover(selected_population)
            crossed_population = [self.response_to_individual(ind["code"], i, iter_dir) 
                                for i, ind in enumerate(crossed_population)]
            self.population = self.evaluate_population(crossed_population, iter_dir)
            self.update_iter()
            
            # Check again before mutation
            if self.function_evals - start_fe >= self.cfg.max_fe:
                logging.info(f"Reached maximum function evaluations ({self.cfg.max_fe}) after crossover")
                break
            
            # Mutation with iteration directory
            mutated_population = self.mutate(iter_dir)
            self.population.extend(self.evaluate_population(mutated_population, iter_dir))
            self.update_iter()
            
            iteration += 1
        
        # Collect results
        result = {
            'best_fitness': self.best_obj_overall,
            'best_code': self.best_code_overall,
            'best_code_path': self.best_code_path_overall,
            'iterations': iteration,
            'function_evals': self.function_evals - start_fe,  # Only count FEs for this description
            'population_history': self.hist_population  # Use filtered history
        }
        
        return result

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual

    def _log_obj_time(self):
        """Log objective values and timing information"""
        current_time = time.time() - self.start_time
        
        # Calculate top 5 average if we have enough individuals
        top_5_avg = float('inf')
        if len(self.best_individuals_overall) >= 5:
            top_5_avg = sum(ind["obj"] for ind in self.best_individuals_overall[:5]) / 5
        
        # Get current best objective
        current_best = self.best_obj_overall if self.best_obj_overall is not None else float('inf')
        
        # Write to log file
        with open(self.obj_time_log_path, 'a') as f:
            f.write(f"{current_time:.2f}\t{current_best:.6f}\t{top_5_avg:.6f}\n")

    def evaluate_population(self, population: list[dict], gen_dir: Path) -> list[dict]:
        """Evaluate population using hybrid approach with updated logging"""
        if len(population) == 0:
            return []
        
        # For seed function or if no predictor available, evaluate all in parallel
        if self.elitist is None or self.performance_predictor is None:
            evaluated = self._evaluate_parallel(population, gen_dir)
            return evaluated
        
        # Get embeddings for all candidates (this is relatively fast)
        # Use 5th best global as threshold instead of elitist
        if len(self.best_individuals_overall) >= 5:
            threshold_individual = self.best_individuals_overall[4]
            threshold_embedding = self.get_code_embedding(threshold_individual["code"])
        else:
            threshold_embedding = self.get_code_embedding(self.elitist["code"])
        
        if threshold_embedding is None:
            evaluated = self._evaluate_parallel(population, gen_dir)
            return evaluated
        
        # Quick pre-filtering using predictor
        to_evaluate = []
        filtered_out = []
        predictions = []
        
        # Get predictions for all candidates
        for individual in population:
            individual["embedding"] = self.get_code_embedding(individual["code"])
            if individual["embedding"] is None:
                to_evaluate.append(individual)
                continue
            
            try:
                prediction = self.performance_predictor.predict_pair(
                    individual["embedding"],
                    threshold_embedding
                )
                predictions.append((individual, prediction['is_code1_better'], prediction['confidence']))
            except Exception as e:
                logging.error(f"Error in prediction: {e}")
                to_evaluate.append(individual)
                continue
        
        # Sort predictions by prediction result and confidence
        predictions.sort(key=lambda x: (-int(x[1]), -x[2]))  # Sort by is_better and confidence
        
        # Determine number of candidates to evaluate based on initial population or eval_ratio
        eval_ratio = 0.5 if self.iteration == 1 else self.cfg.eval_ratio
        eval_count = max(1, int(len(population) * eval_ratio))
        logging.info(f"Will evaluate {eval_count} out of {len(population)} candidates based on {'initial 0.5 ratio' if self.iteration == 1 else f'eval_ratio {self.cfg.eval_ratio}'}")
        
        # Select candidates for evaluation
        for i, (individual, is_better, confidence) in enumerate(predictions):
            if i < eval_count:
                individual["prediction_score"] = confidence
                to_evaluate.append(individual)
                logging.info(f"Selected for evaluation: candidate {i+1} (confidence: {confidence:.4f})")
            else:
                # Count filtered individuals in function_evals
                self.function_evals += 1
                individual = self.mark_invalid_individual(
                    individual,
                    f"Filtered by predictor (rank: {i+1}, confidence: {confidence:.4f})"
                )
                filtered_out.append(individual)
                logging.info(f"Code {individual['response_id']} filtered out by predictor")
        
        # Evaluate remaining candidates in parallel
        if to_evaluate:
            evaluated = self._evaluate_parallel(to_evaluate, gen_dir)
            return evaluated + filtered_out
        return filtered_out

    def _update_best_individuals(self, individual: dict) -> None:
        """Helper method to update best individuals list and log objective values"""
        if not individual.get("exec_success", False):
            return
        
        self.best_individuals_overall.append(individual)
        self.best_individuals_overall.sort(key=lambda x: x.get("obj", float("inf")))
        self.best_individuals_overall = self.best_individuals_overall[:5]
        
        # Update best_obj_overall if needed
        if self.best_obj_overall is None or individual["obj"] < self.best_obj_overall:
            self.best_obj_overall = individual["obj"]
            self.best_code_overall = individual["code"]
            self.best_code_path_overall = individual["code_path"]
        
        self._log_obj_time()  # Log after updating best individuals

    def _load_existing_timing_data(self):
        """Load existing timing data from log files if they exist"""
        for log_path, times_list in [
            (self.api_log_path, self.api_call_times),
            (self.eval_batch_log_path, self.eval_batch_times),
            (self.eval_single_log_path, self.eval_single_times)
        ]:
            if log_path.exists():
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[1:]  # Skip header line
                    times_list.extend([float(line.strip()) for line in lines])

    def _log_timing(self, duration: float, log_type: str):
        """Log timing information by appending to appropriate file"""
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
        
        # Write entire file with updated totals
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Total count: {total_count}, Total time: {total_time:.2f}s\n")
            for t in times_list:
                f.write(f"{t:.4f}\n")

    def _evaluate_parallel(self, population: list[dict], gen_dir: Path) -> list[dict]:
        """Evaluate individuals in parallel with timing"""
        batch_start = time.time()
        
        processes = []
        
        # Launch all processes
        for individual in population:
            if individual["code"] is None:
                individual = self.mark_invalid_individual(individual, "Invalid code")
                processes.append((None, None))
                continue
            
            try:
                single_start = time.time()
                process = self._run_code_async(individual, gen_dir)
                processes.append((process, single_start))
                self.function_evals += 1
            except Exception as e:
                logging.error(f"Failed to launch process: {e}")
                individual = self.mark_invalid_individual(individual, str(e))
                processes.append((None, None))
                continue
        
        # Collect results
        for i, (individual, process_info) in enumerate(zip(population, processes)):
            process, start_time = process_info
            if process is None:
                continue
            
            try:
                process.communicate(timeout=self.cfg.timeout)
                if start_time is not None:
                    single_duration = time.time() - start_time
                    self._log_timing(single_duration, "eval_single")
                
                with open(individual["stdout_filepath"], 'r') as f:
                    stdout_str = f.read()
                traceback_msg = filter_traceback(stdout_str)
                
                if traceback_msg == '':
                    try:
                        individual["obj"] = float(stdout_str.split('\n')[-2]) if self.obj_type == "min" else -float(stdout_str.split('\n')[-2])
                        individual["exec_success"] = True
                        self._update_best_individuals(individual)  # Update best individuals immediately after evaluation
                        
                        # Only add to history if fitness is valid
                        if individual["obj"] != float("inf"):
                            self.hist_population.append(individual.copy())
                            
                        logging.info(f"Iteration {self.iteration}, individual {individual['response_id']}: Objective value: {individual['obj']}")
                    except:
                        individual = self.mark_invalid_individual(individual, "Invalid stdout / objective value")
                else:
                    individual = self.mark_invalid_individual(individual, traceback_msg)
                
            except subprocess.TimeoutExpired:
                logging.info(f"Timeout for individual {individual['response_id']}")
                individual = self.mark_invalid_individual(individual, "Execution timeout")
                process.kill()
            except Exception as e:
                logging.error(f"Error processing results: {e}")
                individual = self.mark_invalid_individual(individual, str(e))
                if process.poll() is None:
                    process.kill()
        
        batch_duration = time.time() - batch_start
        self._log_timing(batch_duration, "eval_batch")
        return population

    def _run_code_async(self, individual: dict, gen_dir: Path) -> subprocess.Popen:
        """Launch code evaluation asynchronously"""
        logging.debug(f"Iteration {self.iteration}: Launching Code Run {individual['response_id']}")
        
        # Write code to both generation directory and problem directory
        code = individual["code"]
        
        # Check if code uses v1 or v2
        uses_v1 = "heuristics_v1" in code
        
        # If code uses v1, write it directly
        # If code uses v2 or neither, ensure it uses v2
        if not uses_v1:
            code = code.replace("heuristics_v1", "heuristics_v2")
            if "heuristics_v2" not in code:
                # If neither v1 nor v2 is present, assume it's v2
                code = code.replace("def heuristics", "def heuristics_v2")
        
        # Write code to files
        with open(individual["code_path"], 'w') as file:
            file.writelines(code + '\n')
        with open(self.output_file, 'w') as file:
            file.writelines(code + '\n')

        with open(individual["stdout_filepath"], 'w') as f:
            eval_file_path = f'{self.root_dir}/problems/{self.problem}/eval.py'
            process = subprocess.Popen(
                ['python', '-u', eval_file_path, f'{self.problem_size}', self.root_dir, "train"],
                stdout=f,
                stderr=f
            )

        block_until_running(individual["stdout_filepath"], log_status=True, 
                           iter_num=self.iteration, response_id=individual['response_id'])
        return process

    def get_dynamic_evaluation_ratio(self) -> float:
        """
        Get dynamic ratio of candidates to evaluate based on predictor success rate
        """
        if not hasattr(self, '_predictor_success_count'):
            self._predictor_success_count = 0
            self._predictor_total_count = 0
            return 0.5  # Start with evaluating 50%
        
        if self._predictor_total_count == 0:
            return 0.5
        
        success_rate = self._predictor_success_count / self._predictor_total_count
        
        # Adjust ratio based on success rate
        if success_rate > 0.8:
            return 0.2  # If predictor is very accurate, evaluate fewer
        elif success_rate > 0.6:
            return 0.3
        elif success_rate > 0.4:
            return 0.4
        else:
            return 0.5  # If predictor is not reliable, evaluate more

    def update_predictor_stats(self, evaluated_individuals: list[dict]) -> None:
        """Update predictor success statistics"""
        if not hasattr(self, '_predictor_success_count'):
            self._predictor_success_count = 0
            self._predictor_total_count = 0
        
        for individual in evaluated_individuals:
            if individual.get("exec_success", False):
                if self.elitist is None or individual["obj"] < self.elitist["obj"]:
                    self._predictor_success_count += 1
            self._predictor_total_count += 1

    def update_iter(self) -> None:
        """
        Update after each iteration
        """
        population = self.population
        objs = [individual["obj"] for individual in population]
        best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))
        
        # update best overall
        if self.best_obj_overall is None or best_obj < self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_sample_idx]["code"]
            self.best_code_path_overall = population[best_sample_idx]["code_path"]
        
        # update elitist
        if self.elitist is None or best_obj < self.elitist["obj"]:
            self.elitist = population[best_sample_idx]
            logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist['obj']}")
        
        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(f"Best obj: {self.best_obj_overall}, Best Code Path: {self.best_code_path_overall}")
        logging.info(f"Function Evals: {self.function_evals}")
        self.iteration += 1
        
    def rank_select(self, population: list[dict]) -> list[dict]:
        """
        Rank-based selection, select individuals with probability proportional to their rank.
        """
        if self.problem_type == "black_box":
            population = [individual for individual in population if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]]
        else:
            population = [individual for individual in population if individual["exec_success"]]
        if len(population) < 2:
            return None
        # Sort population by objective value
        population = sorted(population, key=lambda x: x["obj"])
        ranks = [i for i in range(len(population))]
        probs = [1 / (rank + 1 + len(population)) for rank in ranks]
        # Normalize probabilities
        probs = [prob / sum(probs) for prob in probs]
        selected_population = []
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False, p=probs)
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population
    
    def random_select(self, population: list[dict]) -> list[dict]:
        """
        Random selection, select individuals with equal probability.
        """
        selected_population = []
        # Eliminate invalid individuals
        if self.problem_type == "black_box":
            population = [individual for individual in population if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]]
        else:
            population = [individual for individual in population if individual["exec_success"]]
        if len(population) < 2:
            return None
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False)
            # If two parents have the same objective value, consider them as identical; otherwise, add them to the selected population
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def gen_short_term_reflection_prompt(self, ind1: dict, ind2: dict) -> tuple[list[dict], str, str]:
        """
        Short-term reflection before crossovering two individuals.
        """
        if ind1["obj"] == ind2["obj"]:
            print(ind1["code"], ind2["code"])
            raise ValueError("Two individuals to crossover have the same objective value!")
        # Determine which individual is better or worse
        if ind1["obj"] < ind2["obj"]:
            better_ind, worse_ind = ind1, ind2
        elif ind1["obj"] > ind2["obj"]:
            better_ind, worse_ind = ind2, ind1

        worse_code = filter_code(worse_ind["code"])
        better_code = filter_code(better_ind["code"])
        
        system = self.system_reflector_prompt
        user = self.user_reflector_st_prompt.format(
            func_name = self.func_name,
            func_desc = self.func_desc,
            problem_desc = self.problem_desc,
            worse_code=worse_code,
            better_code=better_code
            )
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # Print reflection prompt for the first iteration
        if self.print_short_term_reflection_prompt:
                logging.info("Short-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_short_term_reflection_prompt = False
        return message, worse_code, better_code

    def short_term_reflection(self, population: list[dict]) -> tuple[list[list[dict]], list[str], list[str]]:
        """Short-term reflection with timing"""
        start_time = time.time()
        
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        for i in range(0, len(population), 2):
            # Select two individuals
            parent_1 = population[i]
            parent_2 = population[i+1]
            
            # Short-term reflection
            messages, worse_code, better_code = self.gen_short_term_reflection_prompt(parent_1, parent_2)
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)
        
        # Asynchronously generate responses
        response_lst = self.reflector_llm.multi_chat_completion(messages_lst)
        
        api_duration = time.time() - start_time
        self._log_timing(api_duration, "api")
        return response_lst, worse_code_lst, better_code_lst
    
    def reflect_on_descriptions(self, better_desc: dict, worse_desc: dict) -> str:
        """Generate reflection comparing two descriptions"""
        start_time = time.time()
        
        system = self.system_reflector_prompt
        user = self.user_reflector_st_prompt.format(
            func_name=self.func_name,
            func_desc=self.func_desc,
            problem_desc=self.problem_desc,
            worse_code=worse_desc["code"],
            better_code=better_desc["code"]
        )
        
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # Print reflection prompt for the first iteration
        if self.print_short_term_reflection_prompt:
            logging.info("Description Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_short_term_reflection_prompt = False
        
        reflection = self.reflector_llm.multi_chat_completion([messages])[0]
        
        # Save description reflection to file
        reflection_dir = self.output_dir / "description_reflections"
        reflection_dir.mkdir(exist_ok=True)
        
        file_name = reflection_dir / f"desc_reflection_iter{self.iteration}.txt"
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(f"Better Description (fitness: {better_desc['obj']}):\n")
            f.write(f"{better_desc['description']}\n\n")
            f.write(f"Worse Description (fitness: {worse_desc['obj']}):\n")
            f.write(f"{worse_desc['description']}\n\n")
            f.write("Reflection:\n")
            f.write(reflection)
        
        api_duration = time.time() - start_time
        self._log_timing(api_duration, "api")
        
        return reflection

    def long_term_reflection(self, short_term_reflections: list[str]) -> None:
        """Long-term reflection with timing"""
        start_time = time.time()
        
        system = self.system_reflector_prompt
        user = self.user_reflector_lt_prompt.format(
            problem_desc = self.problem_desc,
            prior_reflection = self.long_term_reflection_str,
            new_reflection = "\n".join(short_term_reflections),
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        if self.print_long_term_reflection_prompt:
            logging.info("Long-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_long_term_reflection_prompt = False
        
        self.long_term_reflection_str = self.reflector_llm.multi_chat_completion([messages])[0]
        
        # Create reflection directories
        reflection_dir = self.output_dir / "reflections"
        reflection_dir.mkdir(exist_ok=True)
        
        desc_reflection_dir = reflection_dir / "description"
        code_reflection_dir = reflection_dir / "code"
        desc_reflection_dir.mkdir(exist_ok=True)
        code_reflection_dir.mkdir(exist_ok=True)
        
        # Determine if this is a description or code reflection based on content
        is_description_reflection = any("description" in refl.lower() for refl in short_term_reflections)
        target_dir = desc_reflection_dir if is_description_reflection else code_reflection_dir
        
        # Write short-term reflections
        st_file_name = target_dir / f"short_term_reflections_iter{self.iteration}.txt"
        with open(st_file_name, 'w', encoding='utf-8') as f:
            f.write("Short-term Reflections:\n")
            f.write("\n".join(short_term_reflections) + '\n')
        
        # Write long-term reflection
        lt_file_name = target_dir / f"long_term_reflection_iter{self.iteration}.txt"
        with open(lt_file_name, 'w', encoding='utf-8') as f:
            f.write("Long-term Reflection:\n")
            f.write(self.long_term_reflection_str + '\n')
        
        # Write combined reflection history
        history_file = reflection_dir / "reflection_history.txt"
        with open(history_file, 'a', encoding='utf-8') as f:
            f.write(f"\n=== Iteration {self.iteration} ===\n")
            f.write(f"Type: {'Description' if is_description_reflection else 'Code'} Evolution\n")
            f.write("Short-term Reflections:\n")
            f.write("\n".join(short_term_reflections) + '\n\n')
            f.write("Long-term Reflection:\n")
            f.write(self.long_term_reflection_str + '\n')
            f.write("="*50 + '\n')
        
        api_duration = time.time() - start_time
        self._log_timing(api_duration, "api")

    def filter_code(self, code: str) -> str:
        """Filter and clean code for reflection"""
        try:
            # Remove comments
            lines = code.split('\n')
            filtered_lines = [line for line in lines if not line.strip().startswith('#')]
            
            # Remove empty lines at start and end
            while filtered_lines and not filtered_lines[0].strip():
                filtered_lines.pop(0)
            while filtered_lines and not filtered_lines[-1].strip():
                filtered_lines.pop()
            
            return '\n'.join(filtered_lines)
        except Exception as e:
            logging.error(f"Error filtering code: {e}")
            return code

    def crossover_descriptions(self, better_desc: dict, worse_desc: dict) -> str:
        """Crossover two descriptions with timing"""
        start_time = time.time()
        
        system = self.system_generator_prompt
        user = f"""Given two problem descriptions with their performance metrics, please generate a new improved description 
        that combines the strengths of both descriptions while addressing their weaknesses.

        Better performing description (fitness: {better_desc['obj']}):
        {better_desc['description']}

        Worse performing description (fitness: {worse_desc['obj']}):
        {worse_desc['description']}

        Previous reflections and insights:
        {self.long_term_reflection_str}

        Please generate a new description that:
        1. Maintains clarity and completeness
        2. Incorporates successful elements from the better description
        3. Avoids problematic elements from the worse description
        4. Adds any missing important details or constraints

        Provide only the new description without any explanations or additional text.
        """
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        # Generate new description
        response = self.generator_llm.multi_chat_completion([messages])[0]
        
        api_duration = time.time() - start_time
        self._log_timing(api_duration, "api")
        
        # Clean up response - remove any markdown formatting or extra text
        new_desc = response.strip().replace('```', '').strip()
        
        return new_desc

    def mutate(self, gen_dir: Path) -> list[dict]:
        """Elitist-based mutation with timing"""
        start_time = time.time()
        
        system = self.system_generator_prompt
        func_signature1 = self.func_signature.format(version=1) 
        user = self.mutataion_prompt.format(
            user_generator = self.user_generator_prompt,
            reflection = self.long_term_reflection_str + self.external_knowledge,
            func_signature1 = func_signature1,
            elitist_code = filter_code(self.elitist["code"]),
            func_name = self.func_name,
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.print_mutate_prompt:
            logging.info("Mutation Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_mutate_prompt = False
        responses = self.generator_llm.multi_chat_completion([messages], int(self.cfg.pop_size * self.mutation_rate))
        population = [self.response_to_individual(response, response_id, gen_dir) for response_id, response in enumerate(responses)]
        
        api_duration = time.time() - start_time
        self._log_timing(api_duration, "api")
        return population

    def evolve(self) -> Tuple[str, float, str]:
        """Main evolution loop combining description and code evolution"""
        logging.info("Starting MoH evolution...")
        
        current_desc = self.initial_desc
        best_desc = current_desc
        best_fitness = float('inf')
        best_code = None
        
        # Track reflections for each generation
        short_term_reflections = []
        
        # Initialize global results tracking
        global_results = {
            'generations': [],
            'fitness_seen': {}
        }
        
        # Track unique fitness values and their associated descriptions/codes
        unique_solutions = {}  # {fitness: (description, code)}
        
        gen = 0
        while gen < self.num_generations:
            logging.info(f"Generation {gen + 1}/{self.num_generations}")
            self.iteration = gen
            
            # Create generation directory
            gen_dir = self.output_dir / f"generation_{gen}"
            gen_dir.mkdir(exist_ok=True)
            
            # Update problem description
            self.problem_config['description'] = current_desc
            
            # Run code evolution for current description
            reevo_result = self.run_code_evolution(current_desc, gen_dir)
            fitness, code = reevo_result['best_fitness'], reevo_result['best_code']
            
            # Save generation results
            self.save_generation_results(gen, gen_dir, current_desc, reevo_result)
            
            # Update best solution if better
            if fitness < best_fitness:
                best_desc = current_desc
                best_fitness = fitness
                best_code = code
            
            # Track generation in global results
            current_fitness = float(fitness)
            generation_result = {
                'generation': gen,
                'description': current_desc,
                'best_fitness': current_fitness,
                'best_code': code,
                'reevo_results': reevo_result
            }
            global_results['generations'].append(generation_result)
            
            # Track unique fitness values with their descriptions and codes
            if fitness != float('inf'):
                unique_solutions[fitness] = (current_desc, code)
                logging.info(f"Found unique solution {len(unique_solutions)}/12 with fitness {fitness}")
                
                # Check if we have enough unique solutions
                if len(unique_solutions) >= 12:
                    logging.info("Found 12 unique fitness values - stopping evolution")
                    break
            
            # Generate next description using reflection
            if gen < self.num_generations - 1:
                current_desc = self.evolve_description(global_results, gen, short_term_reflections)
            
            gen += 1
        
        # Save unique solutions to a separate file
        unique_solutions_path = self.output_dir / 'unique_solutions.json'
        unique_solutions_list = [
            {
                'fitness': float(fitness),  # Convert numpy float to native float
                'description': desc,
                'code': code
            }
            for fitness, (desc, code) in sorted(unique_solutions.items())
        ]
        
        with open(unique_solutions_path, 'w', encoding='utf-8') as f:
            json.dump(unique_solutions_list, f, indent=2, ensure_ascii=False)
        
        # Save final global results
        with open(self.output_dir / 'global_results.json', 'w', encoding='utf-8') as f:
            json.dump(global_results, f, indent=2, ensure_ascii=False)
        
        # Update original YAML with best description
        self.problem_config['description'] = best_desc
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.problem_config, f, allow_unicode=True)
        
        # Update seed function with best code
        seed_func_path = f"{self.root_dir}/prompts/{self.problem}/seed_func.txt"
        best_code_v1 = best_code.replace("heuristics_v2", "heuristics_v1")
        with open(seed_func_path, 'w', encoding='utf-8') as f:
            f.write(best_code_v1)
        
        logging.info(f"Evolution completed with {len(unique_solutions)} unique solutions")
        logging.info(f"Best fitness achieved: {best_fitness}")
        
        return best_desc, best_fitness, best_code

    def reflect_on_descriptions(self, better_desc: dict, worse_desc: dict) -> str:
        """Generate reflection comparing two descriptions"""
        start_time = time.time()
        
        system = self.system_reflector_prompt
        user = self.user_reflector_st_prompt.format(
            func_name=self.func_name,
            func_desc=self.func_desc,
            problem_desc=self.problem_desc,
            worse_code=worse_desc["code"],
            better_code=better_desc["code"]
        )
        
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # Print reflection prompt for the first iteration
        if self.print_short_term_reflection_prompt:
            logging.info("Description Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_short_term_reflection_prompt = False
        
        reflection = self.reflector_llm.multi_chat_completion([messages])[0]
        
        # Save description reflection to file
        reflection_dir = self.output_dir / "description_reflections"
        reflection_dir.mkdir(exist_ok=True)
        
        file_name = reflection_dir / f"desc_reflection_iter{self.iteration}.txt"
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(f"Better Description (fitness: {better_desc['obj']}):\n")
            f.write(f"{better_desc['description']}\n\n")
            f.write(f"Worse Description (fitness: {worse_desc['obj']}):\n")
            f.write(f"{worse_desc['description']}\n\n")
            f.write("Reflection:\n")
            f.write(reflection)
        
        api_duration = time.time() - start_time
        self._log_timing(api_duration, "api")
        
        return reflection

    def evolve_description(self, global_results: Dict, gen: int, short_term_reflections: List[str]) -> str:
        """Evolve description using reflection-guided process"""
        prev_gens = global_results['generations']
        if len(prev_gens) >= 2:
            # Sort by fitness for selection
            prev_gens.sort(key=lambda x: x['best_fitness'])
            
            # Select two different generations for reflection
            better_gen = prev_gens[0]
            worse_gen = prev_gens[-1]
            
            # Create description objects
            better_desc = {
                "description": better_gen['description'],
                "obj": better_gen['best_fitness'],
                "code": better_gen['best_code']
            }
            worse_desc = {
                "description": worse_gen['description'],
                "obj": worse_gen['best_fitness'],
                "code": worse_gen['best_code']
            }
            
            # Generate reflection using the new method
            reflection = self.reflect_on_descriptions(better_desc, worse_desc)
            short_term_reflections.append(reflection)
            
            # Periodic long-term reflection
            if len(short_term_reflections) >= 3:
                self.long_term_reflection(short_term_reflections)
                short_term_reflections.clear()
            
            # Generate new description
            new_desc = self.crossover_descriptions(better_desc, worse_desc)
            
            # Occasionally mutate
            if random.random() < self.mutation_rate:
                variations = self.generate_variations(new_desc)
                if variations:
                    new_desc = variations[0]
            
            return new_desc
        else:
            # For early generations, just mutate current description
            variations = self.generate_variations(prev_gens[0]['description'])
            return variations[0] if variations else prev_gens[0]['description']

    def get_code_embedding(self, code: str) -> List[float]:
        """Get embedding vector for code"""
        try:
            from utils.get_embeddings import llm_embedding_api
            return llm_embedding_api(
                self.cfg.embedding_llm_model,
                self.cfg.embedding_end_point,
                self.cfg.embedding_api_key,
                code
            )
        except Exception as e:
            logging.error(f"Error getting code embedding: {e}")
            return None

    def crossover(self, selected_population: list[dict]) -> list[dict]:
        """Crossover with timing"""
        start_time = time.time()
        
        messages_lst = []
        for i in range(0, len(selected_population), 2):
            # Select two parents
            parent_1 = selected_population[i]
            parent_2 = selected_population[i+1]
            
            # Determine better and worse parent
            if parent_1["obj"] < parent_2["obj"]:
                better_code = self.filter_code(parent_1["code"])
                worse_code = self.filter_code(parent_2["code"])
            else:
                better_code = self.filter_code(parent_2["code"])
                worse_code = self.filter_code(parent_1["code"])
            
            # Generate crossover prompt
            system = self.system_generator_prompt
            func_signature0 = self.func_signature.format(version=0)
            func_signature1 = self.func_signature.format(version=1)
            user = self.crossover_prompt.format(
                user_generator = self.user_generator_prompt,
                func_signature0 = func_signature0,
                func_signature1 = func_signature1,
                worse_code = worse_code,
                better_code = better_code,
                reflection = self.long_term_reflection_str,
                func_name = self.func_name,
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            messages_lst.append(messages)
            
            # Print crossover prompt for the first iteration
            if self.print_crossover_prompt:
                logging.info("Crossover Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_crossover_prompt = False
        
        # Asynchronously generate responses
        response_lst = self.generator_llm.multi_chat_completion(messages_lst)
        crossed_population = []
        
        # Process each response
        for response_id, response in enumerate(response_lst):
            code = extract_code_from_generator(response)
            if code:
                crossed_population.append({
                    "code": code,
                    "response_id": response_id
                })
        
        assert len(crossed_population) == self.cfg.pop_size, f"Expected {self.cfg.pop_size} individuals, got {len(crossed_population)}"
        
        api_duration = time.time() - start_time
        self._log_timing(api_duration, "api")
        return crossed_population

    def save_generation_results(self, generation: int, gen_dir: Path, description: str, reevo_result: Dict) -> None:
        """Save results for current generation"""
        gen_dir.mkdir(exist_ok=True)

        # Extract results from reevo
        population_history = reevo_result.get('population_history', [])
        
        # Separate successful and failed solutions
        successful_solutions = []
        failed_solutions = []

        for individual in population_history:
            solution = {
                'code': individual["code"],
                'fitness': float(individual.get("obj", float('inf'))),
                'execution_success': individual.get("exec_success", False),
                'error': individual.get("error", None),
                'stdout_file': individual.get("stdout_filepath", None),
                'code_file': individual.get("code_path", None),
                'prediction_score': individual.get("prediction_score", None)
            }

            if individual.get("exec_success", False):
                successful_solutions.append(solution)
            else:
                failed_solutions.append(solution)

        # Sort successful solutions by fitness
        successful_solutions.sort(key=lambda x: x['fitness'])

        # Save generation summary
        generation_summary = {
            'generation': generation,
            'description': description,
            'best_fitness': reevo_result['best_fitness'],
            'best_code': reevo_result['best_code'],
            'best_code_path': reevo_result['best_code_path'],
            'iterations': reevo_result['iterations'],
            'function_evals': reevo_result['function_evals'],
            'successful_solutions_count': len(successful_solutions),
            'failed_solutions_count': len(failed_solutions)
        }

        # Save successful solutions to results.json
        results = {
            'generation': generation,
            'description': description,
            'summary': generation_summary,
            'solutions': successful_solutions
        }

        with open(gen_dir / 'results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save failed solutions to results_err.json
        error_results = {
            'generation': generation,
            'description': description,
            'summary': generation_summary,
            'failed_solutions': failed_solutions
        }

        with open(gen_dir / 'results_err.json', 'w', encoding='utf-8') as f:
            json.dump(error_results, f, indent=2, ensure_ascii=False)

        # Save best solution separately
        with open(gen_dir / 'best_solution.txt', 'w', encoding='utf-8') as f:
            f.write(f"Generation: {generation}\n\n")
            f.write(f"Description:\n{description}\n\n")
            f.write(f"Best Fitness: {reevo_result['best_fitness']}\n\n")
            f.write(f"Best Code:\n{reevo_result['best_code']}\n")

        # Save evolution details
        evo_dir = gen_dir / "evolution"
        if evo_dir.exists():
            with open(gen_dir / 'evolution_summary.txt', 'w', encoding='utf-8') as f:
                f.write(f"Total iterations: {reevo_result['iterations']}\n")
                f.write(f"Total function evaluations: {reevo_result['function_evals']}\n")
                f.write(f"Final best fitness: {reevo_result['best_fitness']}\n")

    def generate_variations(self, description: str) -> List[str]:
        """Generate variations of a description using LLM"""
        start_time = time.time()
        
        # Create prompt for generating variations
        system = self.system_generator_prompt
        user = f"""Please generate a variation of the following problem description.
        Keep the core requirements but express them differently or add more details.
        
        Original description:
        {description}
        
        Consider the following aspects when generating variations:
        1. Clarity: Make sure the requirements are clearly stated
        2. Completeness: Include all necessary constraints and conditions
        3. Precision: Use specific and unambiguous language
        4. Context: Add relevant domain context if helpful
        
        Please provide only the new description, without any explanations or additional text.
        """
        
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        try:
            # Generate 1-3 variations
            num_variations = random.randint(1, 3)
            responses = self.generator_llm.multi_chat_completion([messages], num_variations)
            
            # Clean up responses
            variations = []
            for response in responses:
                # Remove any markdown formatting or extra whitespace
                cleaned = response.strip().replace('```', '').strip()
                if cleaned:
                    variations.append(cleaned)
            
            # Write variations to file for tracking
            gen_dir = self.output_dir / f"generation_{self.iteration}"
            gen_dir.mkdir(exist_ok=True)
            
            with open(gen_dir / 'description_variations.txt', 'w', encoding='utf-8') as f:
                for i, var in enumerate(variations):
                    f.write(f"Variation {i+1}:\n{var}\n\n")
            
            api_duration = time.time() - start_time
            self._log_timing(api_duration, "api")
            
            return variations
        
        except Exception as e:
            logging.error(f"Error generating description variations: {e}")
            return []
