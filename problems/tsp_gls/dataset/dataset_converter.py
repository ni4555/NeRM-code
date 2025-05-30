# -*- coding: gbk -*-
import pickle
import numpy as np

def convert_tsp_to_npy():
    # ��ȡԭʼ��TSP500.pkl�ļ�
    with open('TSP1000.pkl', 'rb') as f:
        tsp_data = pickle.load(f)
    
    # ��ȡ������Ϣ
    # ����tsp_data�е�ÿ��ʵ��������coordinates�ֶ�
    corrdinates_list = tsp_data['coordinate']
    
    # ת��Ϊnumpy����
    coordinates_array = np.array(corrdinates_list[:64])
    
    # ����Ϊ.npy��ʽ
    np.save('train1000_dataset.npy', coordinates_array)
    
    print(f"ת����ɣ�������״: {coordinates_array.shape}")


def optimal_tracker(pkl_list, samples_num=64):
    results = ""
    for inst_name in pkl_list:
        # ��ȡԭʼ��TSP500.pkl�ļ�
        with open(inst_name + '.pkl', 'rb') as f:
            tsp_data = pickle.load(f)

        # ��ȡ������Ϣ
        # ����tsp_data�е�ÿ��ʵ��������coordinates�ֶ�
        opt_cost = tsp_data['cost']

        sample_opt_cost = opt_cost[:samples_num]
        results += f"{inst_name}: {np.mean(sample_opt_cost)}\n"

    with open('optimal_tracker.txt', 'w') as f:
        f.write(results)


if __name__ == "__main__":
    # convert_tsp_to_npy()

    samples_num = 64
    pkl_list = ['TSP20', 'TSP50', 'TSP100', 'TSP200', 'TSP500', 'TSP1000']
    optimal_tracker(pkl_list, samples_num)


