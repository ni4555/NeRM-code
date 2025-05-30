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


class ReEvo:
    def __init__(
        self, 
        cfg: DictConfig, 
        root_dir: str, 
        generator_llm: BaseClient, 
        reflector_llm: Optional[BaseClient] = None,
    ) -> None:
        # Initialize timing attributes first, before anything else
        self.api_log_path = Path("api_invoke_log.log")
        self.eval_batch_log_path = Path("eval_time_log_batch.log")
        self.eval_single_log_path = Path("eval_time_log_single.log")
        self.obj_time_log_path = Path("obj_time.log")  # New log file path
        
        # Initialize timing lists
        self.api_call_times = []
        self.eval_batch_times = []
        self.eval_single_times = []
        self.start_time = time.time()  # Record start time
        
        # Load existing timing data if files exist
        self._load_existing_timing_data()
        
        # Continue with rest of initialization
        self.cfg = cfg
        self.generator_llm = generator_llm
        self.reflector_llm = reflector_llm or generator_llm
        self.root_dir = root_dir
        
        self.mutation_rate = cfg.mutation_rate
        self.iteration = 0
        self.function_evals = 0
        self.elitist = None
        self.long_term_reflection_str = ""
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None
        self.best_individuals_overall = []  # Add this line to track best individuals
        
        self.init_prompt()
        self.init_population()

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


    def init_population(self) -> None:
        # Evaluate the seed function, and set it as Elite
        logging.info("Evaluating seed function...")
        code = extract_code_from_generator(self.seed_func).replace("v1", "v2")
        logging.info("Seed function code: \n" + code)
        seed_ind = {
            "stdout_filepath": f"problem_iter{self.iteration}_stdout0.txt",
            "code_path": f"problem_iter{self.iteration}_code0.py",
            "code": code,
            "response_id": 0,
        }
        self.seed_ind = seed_ind
        self.population = self.evaluate_population([seed_ind])

        # If seed function is invalid, stop
        if not self.seed_ind["exec_success"]:
            raise RuntimeError(f"Seed function is invalid. Please check the stdout file in {os.getcwd()}.")

        self.update_iter()
        
        # Generate responses
        system = self.system_generator_prompt
        user = self.user_generator_prompt + "\n" + self.seed_prompt + "\n" + self.long_term_reflection_str
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        logging.info("Initial Population Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)

        responses = self.generator_llm.multi_chat_completion([messages], self.cfg.init_pop_size, temperature = self.generator_llm.temperature + 0.3) # Increase the temperature for diverse initial population
        population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(responses)]

        # Run code and evaluate population
        population = self.evaluate_population(population)

        # Update iteration
        self.population = population
        self.update_iter()

    
    def response_to_individual(self, response: str, response_id: int, file_name: str=None) -> dict:
        """
        Convert response to individual
        """
        # Write response to file
        file_name = f"problem_iter{self.iteration}_response{response_id}.txt" if file_name is None else file_name + ".txt"
        with open(file_name, 'w') as file:
            file.writelines(response + '\n')

        code = extract_code_from_generator(response)

        # Extract code and description from response
        std_out_filepath = f"problem_iter{self.iteration}_stdout{response_id}.txt" if file_name is None else file_name + "_stdout.txt"
        
        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": f"problem_iter{self.iteration}_code{response_id}.py",
            "code": code,
            "response_id": response_id,
        }
        return individual

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual


    def evaluate_population(self, population: list[dict]) -> list[float]:
        """
        Evaluate population by running code in parallel and computing objective values.
        """
        batch_start = time.time()
        
        inner_runs = []
        for response_id, individual in enumerate(population):
            if individual["code"] is None:
                individual = self.mark_invalid_individual(individual, "Invalid code")
                inner_runs.append(None)
                continue
            
            try:
                single_start = time.time()
                process = self._run_code(individual, response_id)
                inner_runs.append((process, single_start))
                self.function_evals += 1
            except Exception as e:
                logging.error(f"Failed to launch process: {e}")
                individual = self.mark_invalid_individual(individual, str(e))
                inner_runs.append(None)
                continue

        # Update population with objective values
        for response_id, run_info in enumerate(inner_runs):
            if run_info is None:
                continue
                
            process, start_time = run_info
            try:
                process.communicate(timeout=self.cfg.timeout)
                single_duration = time.time() - start_time
                self._log_timing(single_duration, "eval_single")
                
                individual = population[response_id]
                stdout_filepath = individual["stdout_filepath"]
                with open(stdout_filepath, 'r') as f:  # read the stdout file
                    stdout_str = f.read() 
                traceback_msg = filter_traceback(stdout_str)
                
                individual = population[response_id]
                # Store objective value for each individual
                if traceback_msg == '': # If execution has no error
                    try:
                        individual["obj"] = float(stdout_str.split('\n')[-2]) if self.obj_type == "min" else -float(stdout_str.split('\n')[-2])
                        individual["exec_success"] = True
                        self._update_best_individuals(individual)  # Update best individuals immediately after evaluation
                    except:
                        population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid std out / objective value!")
                else: # Otherwise, also provide execution traceback error feedback
                    population[response_id] = self.mark_invalid_individual(population[response_id], traceback_msg)

                logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['obj']}")
            except subprocess.TimeoutExpired as e:
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                process.kill()
                continue

        batch_duration = time.time() - batch_start
        self._log_timing(batch_duration, "eval_batch")
        return population


    def _run_code(self, individual: dict, response_id) -> subprocess.Popen:
        """
        Write code into a file and run eval script.
        """
        logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")
        
        with open(self.output_file, 'w') as file:
            file.writelines(individual["code"] + '\n')

        # Execute the python file with flags
        with open(individual["stdout_filepath"], 'w') as f:
            eval_file_path = f'{self.root_dir}/problems/{self.problem}/eval.py' if self.problem_type != "black_box" else f'{self.root_dir}/problems/{self.problem}/eval_black_box.py' 
            process = subprocess.Popen(['python', '-u', eval_file_path, f'{self.problem_size}', self.root_dir, "train"],
                                        stdout=f, stderr=f)

        block_until_running(individual["stdout_filepath"], log_status=True, iter_num=self.iteration, response_id=response_id)
        # time.sleep(4)
        return process

    
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
        """
        Short-term reflection before crossovering two individuals.
        """
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
    
    def long_term_reflection(self, short_term_reflections: list[str]) -> None:
        """
        Long-term reflection before mutation.
        """
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
        
        # Write reflections to file
        file_name = f"problem_iter{self.iteration}_short_term_reflections.txt"
        with open(file_name, 'w') as file:
            file.writelines("\n".join(short_term_reflections) + '\n')
        
        file_name = f"problem_iter{self.iteration}_long_term_reflection.txt"
        with open(file_name, 'w') as file:
            file.writelines(self.long_term_reflection_str + '\n')
        
        api_duration = time.time() - start_time
        self._log_timing(api_duration, "api")


    def crossover(self, short_term_reflection_tuple: tuple[list[list[dict]], list[str], list[str]]) -> list[dict]:
        start_time = time.time()
        
        reflection_content_lst, worse_code_lst, better_code_lst = short_term_reflection_tuple
        messages_lst = []
        for reflection, worse_code, better_code in zip(reflection_content_lst, worse_code_lst, better_code_lst):
            # Crossover
            system = self.system_generator_prompt
            func_signature0 = self.func_signature.format(version=0)
            func_signature1 = self.func_signature.format(version=1)
            user = self.crossover_prompt.format(
                user_generator = self.user_generator_prompt,
                func_signature0 = func_signature0,
                func_signature1 = func_signature1,
                worse_code = worse_code,
                better_code = better_code,
                reflection = reflection,
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
        crossed_population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(response_lst)]

        assert len(crossed_population) == self.cfg.pop_size
        
        api_duration = time.time() - start_time
        self._log_timing(api_duration, "api")
        return crossed_population


    def mutate(self) -> list[dict]:
        """Elitist-based mutation. We only mutate the best individual to generate n_pop new individuals."""
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
        population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(responses)]
        
        api_duration = time.time() - start_time
        self._log_timing(api_duration, "api")
        return population


    def evolve(self):
        while True:
            # Check termination conditions
            if hasattr(self.cfg, 'exp_obj_test_only'):
                # If target objective value is set and reached, terminate
                if self.best_obj_overall is not None and self.best_obj_overall <= self.cfg.exp_obj_test_only:
                    logging.info(f"Reached target objective value: {self.best_obj_overall}")
                    break
            elif self.function_evals >= self.cfg.max_fe:
                # If max function evaluations reached, terminate
                logging.info(f"Reached maximum function evaluations: {self.function_evals}")
                break
                
            # If all individuals are invalid, stop
            if all([not individual["exec_success"] for individual in self.population]):
                raise RuntimeError(f"All individuals are invalid. Please check the stdout files in {os.getcwd()}.")
            
            # Select
            population_to_select = self.population if (self.elitist is None or self.elitist in self.population) else [self.elitist] + self.population
            selected_population = self.random_select(population_to_select)
            if selected_population is None:
                raise RuntimeError("Selection failed. Please check the population.")
                
            # Short-term reflection
            short_term_reflection_tuple = self.short_term_reflection(selected_population)
            
            # Crossover
            crossed_population = self.crossover(short_term_reflection_tuple)
            
            # Evaluate
            self.population = self.evaluate_population(crossed_population)
            
            # Update
            self.update_iter()
            
            # Long-term reflection
            self.long_term_reflection([response for response in short_term_reflection_tuple[0]])
            
            # Mutate
            mutated_population = self.mutate()
            
            # Evaluate
            self.population.extend(self.evaluate_population(mutated_population))
            
            # Update
            self.update_iter()

        return self.best_code_overall, self.best_code_path_overall
