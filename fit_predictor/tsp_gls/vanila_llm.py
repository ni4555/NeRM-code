# -*- coding: gbk -*-
import json
import logging
import random
from datetime import datetime
import os
import numpy as np
from scipy import stats
import openai
from zhipuai import ZhipuAI
import time
from typing import List, Dict, Any, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'./results/llm_predictor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class LLMPredictor:
    def __init__(self, openai_key: str, zhipuai_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_key)
        self.zhipuai_client = ZhipuAI(api_key=zhipuai_key)
        
    def load_test_pairs(self, test_pairs_path: str, code_results_path: str) -> List[Dict]:
        """加载测试数据对"""
        logging.info("Loading test pairs and code data...")
        
        # 加载测试对
        with open(test_pairs_path, 'r') as f:
            test_pairs_data = json.load(f)
            
        # 加载代码数据
        with open(code_results_path, 'r') as f:
            code_data = json.load(f)
            
        # 创建代码ID到代码的映射
        code_dict = {str(item['id']): item['code'] for item in code_data['results']}
        
        # 为测试对添加代码
        test_pairs = []
        for pair in test_pairs_data['test_pairs']:
            test_pairs.append({
                'code1_id': pair['code1_id'],
                'code2_id': pair['code2_id'],
                'code1': code_dict[pair['code1_id']],
                'code2': code_dict[pair['code2_id']],
                'code1_performance': pair['code1_performance'],
                'code2_performance': pair['code2_performance'],
                'is_code1_better': pair['is_code1_better'],
                'performance_diff': pair['performance_diff']
            })
            
        logging.info(f"Loaded {len(test_pairs)} test pairs")
        return test_pairs

    def get_gpt_prediction(self, code1: str, code2: str) -> bool:
        """使用GPT-3.5预测两段代码的性能优劣"""
        prompt = f"""You are an expert in analyzing algorithm performance. Please compare these two implementations of the same algorithm and predict which one will have better performance (lower execution time). Only respond with "1" if you think the first code is better, or "2" if you think the second code is better.

First implementation:
{code1}

Second implementation:
{code2}

Response (1 or 2):"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            prediction = response.choices[0].message.content.strip()
            return prediction == "1"
        except Exception as e:
            logging.error(f"Error in GPT prediction: {e}")
            # 在出错时返回随机预测
            return random.choice([True, False])

    def get_glm_prediction(self, code1: str, code2: str) -> bool:
        """使用GLM-4预测两段代码的性能优劣"""
        prompt = f"""You are an expert in analyzing algorithm performance. Please compare these two implementations of the same algorithm and predict which one will have better performance (lower execution time). Only respond with "1" if you think the first code is better, or "2" if you think the second code is better.

First implementation:
{code1}

Second implementation:
{code2}

Response (1 or 2):"""
        
        try:
            response = self.zhipuai_client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            prediction = response.choices[0].message.content.strip()
            return prediction == "1"
        except Exception as e:
            logging.error(f"Error in GLM prediction: {e}")
            # 在出错时返回随机预测
            return random.choice([True, False])

    def evaluate_model(self, test_pairs: List[Dict], model_name: str) -> Tuple[Dict, List, List]:
        """评估模型性能"""
        logging.info(f"Evaluating {model_name}...")
        
        results = {
            'very_close': {'correct': 0, 'total': 0},  # 0-0.05
            'close': {'correct': 0, 'total': 0},       # 0.05-0.1
            'medium': {'correct': 0, 'total': 0},      # 0.1-0.2
            'far': {'correct': 0, 'total': 0}         # >0.2
        }
        
        predictions = []
        true_labels = []
        
        for pair in test_pairs:
            # 根据模型类型选择预测函数
            if model_name == "gpt-3.5-turbo":
                predicted_better = self.get_gpt_prediction(pair['code1'], pair['code2'])
            else:  # GLM-4
                predicted_better = self.get_glm_prediction(pair['code1'], pair['code2'])
            
            # 添加延迟以避免API限制
            time.sleep(1)
            
            # 根据性能差异分类
            perf_diff = pair['performance_diff']
            if perf_diff < 0.05:
                category = 'very_close'
            elif perf_diff < 0.1:
                category = 'close'
            elif perf_diff < 0.2:
                category = 'medium'
            else:
                category = 'far'
            
            results[category]['total'] += 1
            if predicted_better == pair['is_code1_better']:
                results[category]['correct'] += 1
            
            predictions.append(predicted_better)
            true_labels.append(pair['is_code1_better'])
            
            # 定期输出进度
            if len(predictions) % 10 == 0:
                logging.info(f"Processed {len(predictions)}/{len(test_pairs)} pairs")
        
        return results, predictions, true_labels

    def calculate_correlation_metrics(self, predictions: List[bool], true_labels: List[bool]) -> Dict:
        """计算相关性指标"""
        # 转换为数值型数组
        pred_numeric = np.array(predictions, dtype=int)
        true_numeric = np.array(true_labels, dtype=int)
        
        # 计算相关系数
        kendall_tau, _ = stats.kendalltau(pred_numeric, true_numeric)
        pearson_r, _ = stats.pearsonr(pred_numeric, true_numeric)
        spearman_rho, _ = stats.spearmanr(pred_numeric, true_numeric)
        
        return {
            'kendall_tau': kendall_tau,
            'pearson_r': pearson_r,
            'spearman_rho': spearman_rho
        }

def main():
    # 设置随机种子
    random.seed(42)
    
    # 创建结果目录
    os.makedirs('./results', exist_ok=True)
    
    # 初始化预测器
    predictor = LLMPredictor(
        openai_key="your-openai-key",
        zhipuai_key="ca510b259db802fa9937680a57980bad.1mZkBFQP9vnl4J31"
    )
    
    # 加载测试数据
    test_pairs_path = './results/code_performance_predictor_simple/results/test_pairs_20250105_102116.json'
    code_results_path = 'tsp_gls_results_with_embeddings.json'
    test_pairs = predictor.load_test_pairs(test_pairs_path, code_results_path)
    
    # 评估模型
    # models = ["gpt-3.5-turbo", "glm-4-flash"]
    models = ["glm-4-flash"]
    all_results = {}
    
    for model_name in models:
        logging.info(f"\nEvaluating {model_name}...")
        results, predictions, true_labels = predictor.evaluate_model(test_pairs, model_name)
        
        # 计算相关性指标
        correlation_metrics = predictor.calculate_correlation_metrics(predictions, true_labels)
        
        # 计算总体准确率
        total_correct = sum(p == t for p, t in zip(predictions, true_labels))
        total_accuracy = total_correct / len(predictions)
        
        # 保存结果
        all_results[model_name] = {
            'accuracy': total_accuracy,
            'detailed_results': {
                category: {
                    'correct': stats['correct'],
                    'total': stats['total'],
                    'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                }
                for category, stats in results.items()
            },
            'correlation_metrics': correlation_metrics
        }
        
        # 输出详细结果
        logging.info(f"\n{model_name} Results:")
        logging.info("=" * 50)
        logging.info(f"Overall accuracy: {total_accuracy:.4f}")
        
        logging.info("\nDetailed Results by Category:")
        for category, stats in results.items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                logging.info(f"{category}:")
                logging.info(f"  Total pairs: {stats['total']}")
                logging.info(f"  Correct predictions: {stats['correct']}")
                logging.info(f"  Accuracy: {accuracy:.4f}")
        
        logging.info("\nCorrelation Metrics:")
        logging.info(f"Kendall's Tau: {correlation_metrics['kendall_tau']:.4f}")
        logging.info(f"Pearson's R: {correlation_metrics['pearson_r']:.4f}")
        logging.info(f"Spearman's Rho: {correlation_metrics['spearman_rho']:.4f}")
    
    # 保存所有结果
    results_path = f'./results/llm_evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logging.info(f"\nAll results saved to {results_path}")

if __name__ == "__main__":
    main()
