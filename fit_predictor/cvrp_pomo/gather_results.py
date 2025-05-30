import os
import json
import re
from pathlib import Path

def parse_stdout_file(stdout_path):
    """Parse the stdout file to get results for each instance"""
    try:
        with open(stdout_path, 'r') as f:
            lines = f.readlines()
            
        # Check if there are error messages
        error_indicators = ['error', 'exception', 'traceback']
        for line in lines:
            if any(indicator in line.lower() for indicator in error_indicators):
                return None
        
        # Find the average value after "[*] Average:"
        for i, line in enumerate(lines):
            if line.strip() == '[*] Average:':
                if i + 1 < len(lines):  # Make sure there's a next line
                    try:
                        value = float(lines[i + 1].strip())
                        return [value]  # Return single value in a list
                    except:
                        return None
                        
        return None
        
    except Exception as e:
        return None

def get_code_content(code_path):
    """Read code content from file and remove markdown code block markers if present"""
    try:
        with open(code_path, 'r') as f:
            content = f.read()
            
        # Remove markdown code block markers if present
        if content.startswith('```python\n'):
            content = content[len('```python\n'):]
        if content.endswith('\n```'):
            content = content[:-4]
        elif content.endswith('```'):
            content = content[:-3]
            
        return content.strip()
    except:
        return None

def gather_results(base_path):
    """Gather all valid results from the evaluation directory"""
    results = []
    base_path = Path(base_path)
    code_id = 0  # Initialize counter for code IDs
    
    # Walk through all subdirectories
    for run_dir in base_path.glob("run*"):
        for date_dir in run_dir.glob("????-??-??_??-??-??"):
            
            # Handle regular code files
            for code_file in date_dir.glob("problem_iter*_response*.txt"):
                stdout_file = Path(str(code_file) + "_stdout.txt")
                if stdout_file.exists():
                    instance_results = parse_stdout_file(stdout_file)
                    if instance_results:
                        code_content = get_code_content(code_file)
                        if code_content:
                            results.append({
                                "id": code_id,
                                "code": code_content,
                                "results": instance_results,
                                "path": str(code_file.relative_to(base_path))
                            })
                            code_id += 1
            
            # Handle coevolve results
            coevolve_dir = date_dir / "coevolve"
            if coevolve_dir.exists():
                for gen_dir in coevolve_dir.glob("generation_*"):
                    for code_file in gen_dir.glob("code_*.py"):
                        stdout_file = gen_dir / f"stdout_{code_file.stem.split('_')[1]}.txt"
                        if stdout_file.exists():
                            instance_results = parse_stdout_file(stdout_file)
                            if instance_results:
                                code_content = get_code_content(code_file)
                                if code_content:
                                    results.append({
                                        "id": code_id,
                                        "code": code_content,
                                        "results": instance_results,
                                        "path": str(code_file.relative_to(base_path))
                                    })
                                    code_id += 1
    
    return results

def main():
    base_path = "./evaluates/"
    results = gather_results(base_path)
    
    # Save results to JSON file
    output_file = "cvrp_pomo_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "total_valid_codes": len(results),
            "results": results
        }, f, indent=2)
    
    print(f"Gathered {len(results)} valid results and saved to {output_file}")

if __name__ == "__main__":
    main()
