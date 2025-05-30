import os
import numpy as np
from openai import OpenAI
import json
import pickle

def load_json_file(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

def load_pkl_file(pkl_file_path):
    with open(pkl_file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# get embedding of text
def llm_embedding_api(llm_model, end_point, api_key, text_content):
    client = OpenAI(
        api_key=api_key,
        base_url=end_point
    )

    response = client.embeddings.create(
        input=[text_content],
        model=llm_model,
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "NONE"}
    )

    response = response.data[0].embedding
    return response

def batch_fetch_instance_embeddings(instances, llm_model, end_point, api_key):
    """Get embeddings for TSP instances"""
    instances_with_embeddings = []
    
    for instance in instances:
        # Convert coordinates to string representation for embedding
        coords_str = np.array2string(instance, precision=8, separator=',')
        text_embedding = llm_embedding_api(llm_model, end_point, api_key, coords_str)
        instances_with_embeddings.append({
            'coordinates': instance.tolist(),
            'embedding': text_embedding
        })
        
    return instances_with_embeddings

def get_cvrp_pomo_code_embeddings():
    """Get embeddings for TSP-GLS codes"""
    # LLM configuration
    llm_model = 'text-embedding-ada-002'
    end_point = ''  # e.g. https://api.bianxie.ai/v1
    api_key = ''  # e.g. sk-xxxxxxxxxx
    
    # Load TSP-GLS results
    results_path = './cvrp_pomo_results.json'
    results_data = load_json_file(results_path)
    
    # Get embeddings for all codes
    codes_with_embeddings = []
    for code_data in results_data['results']:
        print("fetching embedding for code: ", code_data['id'])
        code_embedding = llm_embedding_api(llm_model, end_point, api_key, code_data['code'])
        codes_with_embeddings.append({
            'id': code_data['id'],
            'code': code_data['code'],
            'results': code_data['results'],
            'path': code_data['path'],
            'embedding': code_embedding  # Convert numpy array to list for JSON serialization
        })
    
    # Save results as JSON
    output_path = './cvrp_pomo_results_with_embeddings.json'
    with open(output_path, 'w') as f:
        json.dump({
            'total_valid_codes': len(codes_with_embeddings),
            'results': codes_with_embeddings
        }, f, indent=2)
    
    print(f"Saved embeddings for {len(codes_with_embeddings)} codes to {output_path}")

if __name__ == "__main__":
    # Get embeddings for both instances and codes
    get_cvrp_pomo_code_embeddings()

