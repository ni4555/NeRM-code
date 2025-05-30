import pickle
import json

def load_json_file(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

def load_pkl_file(pkl_file_path):
    # 打开pkl文件
    with open(pkl_file_path, 'rb') as file:
        # 加载pkl文件
        data = pickle.load(file)
    return data


if __name__ == '__main__':
    base_dir = './'
    pkl_file_path = base_dir + 'train200_dataset_with_embeddings.pkl'
    pkl_data = load_pkl_file(pkl_file_path)

    pkl_data_columns = ['instance_embedding', 'instance_id', 'reference_embeddings', 'reference_ids', 'reference_fitness', 'target_embedding', 'target_id', 'target_fitness', 'actual_rank']

    selected_columns = ['instance_id', 'reference_ids', 'reference_fitness', 'target_id', 'target_fitness', 'actual_rank']
    selected_data = []
    for row_data in pkl_data:
        selected_data.append(row_data[selected_columns])

    print(selected_data)
