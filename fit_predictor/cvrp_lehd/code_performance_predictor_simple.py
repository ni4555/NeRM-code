# -*- coding: gbk -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from datetime import datetime
import os
import pickle
import json
import random
import sys

# ������־
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'./results/predictor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class PerformancePredictor(nn.Module):
    def __init__(self, code_dim=1536):
        super(PerformancePredictor, self).__init__()
        
        # ������ȡ��
        self.feature_extractor = nn.Sequential(
            nn.Linear(code_dim, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.ReLU()
        )
        
        # �Ƚ�����
        self.comparison = nn.Sequential(
            nn.Linear(384, 192),  # 384 = 192 * 2 (�������������ƴ��)
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(192, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # �����������[-1, 1]��Χ��
        )
    
    def forward(self, x1, x2):
        # ��ȡ����
        feat1 = self.feature_extractor(x1)
        feat2 = self.feature_extractor(x2)
        
        # ƴ��������������Ե÷�
        combined = torch.cat([feat1, feat2], dim=1)
        return self.comparison(combined)

class CodePerformancePredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PerformancePredictor().to(self.device)
        # ʹ��BCE��ʧ��������ת��Ϊ������
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.01)
        
    def prepare_training_data(self, code_embeddings, code_performances, test_size=0.2):
        """׼��ѵ�����ݺͲ�������"""
        logging.info("Preparing training and test data...")
        
        # ����ָ����IDΪѵ�����Ͳ��Լ�
        code_ids = list(code_embeddings.keys())
        n_test = int(len(code_ids) * test_size)
        test_code_ids = set(random.sample(code_ids, n_test))
        train_code_ids = set(code_ids) - test_code_ids
        
        # �������ݼ�����
        split_info = {
            'train_code_ids': list(train_code_ids),
            'test_code_ids': list(test_code_ids),
            'split_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        split_path = f'./results/dataset_split_{split_info["split_timestamp"]}.json'
        with open(split_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        # ����ѵ����
        train_codes = list(train_code_ids)
        
        # �������п��ܵĴ���Բ�����ÿ�������ϵ����ܲ���
        potential_pairs = []
        for i, code1_id in enumerate(train_codes):
            perfs1 = code_performances[code1_id]
            for code2_id in train_codes[i+1:]:
                perfs2 = code_performances[code2_id]
                
                # ��ÿ������ֱ�Ƚ�����
                for prob_idx in range(len(perfs1)):
                    perf1 = perfs1[prob_idx]
                    perf2 = perfs2[prob_idx]
                    perf_diff = abs(perf1 - perf2)
                    
                    # ֻ�е����������Բ����Ҳ����ʱ�����뿼��
                    if perf_diff > 0.01 and abs(perf1 - perf2) > 1e-10:
                        potential_pairs.append({
                            'code1_id': code1_id,
                            'code2_id': code2_id,
                            'code1_embedding': code_embeddings[code1_id],
                            'code2_embedding': code_embeddings[code2_id],
                            'code1_performance': perf1,
                            'code2_performance': perf2,
                            'perf_diff': perf_diff,
                            'problem_idx': prob_idx
                        })
        
        # �����ܲ�������
        potential_pairs.sort(key=lambda x: x['perf_diff'])
        
        # ����ʵ�����ֲܷ������ϸ�µ����ܲ���ֽ��
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.5, float('inf')]  # �����ݷֳ�5�����䣺
        # ΢С����(0.01-0.05)��С����(0.05-0.1)���еȲ���(0.1-0.2)����������(0.2-0.5)�������(>0.5)
        bucket_pairs = [[] for _ in range(len(thresholds)-1)]
        
        # �����ݷ��䵽��ͬ��Ͱ��
        for pair in potential_pairs:
            for i, (low, high) in enumerate(zip(thresholds[:-1], thresholds[1:])):
                if low <= pair['perf_diff'] < high:
                    bucket_pairs[i].append(pair)
                    break
        
        # ����ÿ��Ͱ�Ĳ���Ȩ�أ���ע���еȺ��������������
        bucket_weights = [0.3, 0.6, 1.0, 1.5, 2.0]  # Ȩ�ش�С����
        base_samples = 2000  # ��С�����������Ա�����Ȳ���
        
        train_pairs = []
        for bucket, weight in zip(bucket_pairs, bucket_weights):
            if bucket:
                # ����Ȩ�ؼ����������
                sample_size = min(int(base_samples * weight), len(bucket))
                train_pairs.extend(random.sample(bucket, sample_size))
        
        random.shuffle(train_pairs)
        
        # ��¼ѵ�����ݷֲ�
        logging.info("\nTraining data distribution:")
        perf_ranges = [
            (0.01, 0.05, "Minimal diff"),
            (0.05, 0.1, "Small diff"),
            (0.1, 0.2, "Medium diff"),
            (0.2, 0.5, "Significant diff"),
            (0.5, float('inf'), "Large diff")
        ]
        for min_diff, max_diff, label in perf_ranges:
            count = sum(1 for pair in train_pairs if min_diff <= pair['perf_diff'] < max_diff)
            total = sum(1 for pair in potential_pairs if min_diff <= pair['perf_diff'] < max_diff)
            logging.info(f"  {label} [{min_diff:.2f}, {max_diff if max_diff != float('inf') else '��'}): "
                        f"{count} pairs (from {total} total)")
        
        logging.info(f"Final training set size: {len(train_pairs)} pairs")
        
        # ׼����������
        test_pairs = []
        test_codes = list(test_code_ids)
        
        # Ϊ���Լ����ɴ����
        for i, code1_id in enumerate(test_codes):
            perfs1 = code_performances[code1_id]
            for code2_id in test_codes[i+1:]:
                perfs2 = code_performances[code2_id]
                
                for prob_idx in range(len(perfs1)):
                    perf1 = perfs1[prob_idx]
                    perf2 = perfs2[prob_idx]
                    perf_diff = abs(perf1 - perf2)
                    
                    # ֻ�е����������Բ����Ҳ����ʱ�����뿼��
                    if perf_diff > 0.01 and abs(perf1 - perf2) > 1e-10:  # ��Ӳ�����ж�
                        test_pairs.append({
                            'code1_id': code1_id,
                            'code2_id': code2_id,
                            'code1_embedding': code_embeddings[code1_id],
                            'code2_embedding': code_embeddings[code2_id],
                            'code1_performance': perf1,
                            'code2_performance': perf2,
                            'is_code1_better': perf1 < perf2,
                            'performance_diff': perf_diff,
                            'problem_idx': prob_idx
                        })
        
        # ���Ȳ������Զ�
        if len(test_pairs) > 2000:
            test_pairs.sort(key=lambda x: x['performance_diff'])
            sampled_test_pairs = []
            
            # Ϊ���Լ�ʹ�����Ƶķ�Ͱ����
            test_buckets = [[] for _ in range(len(thresholds)-1)]
            for pair in test_pairs:
                for i, (low, high) in enumerate(zip(thresholds[:-1], thresholds[1:])):
                    if low <= pair['performance_diff'] < high:
                        test_buckets[i].append(pair)
                        break
            
            # ��ÿ��Ͱ�в���
            samples_per_bucket = [400, 600, 1000]  # �����������������
            for bucket, target_size in zip(test_buckets, samples_per_bucket):
                if bucket:
                    actual_size = min(target_size, len(bucket))
                    sampled_test_pairs.extend(random.sample(bucket, actual_size))
            
            test_pairs = sampled_test_pairs
        
        random.shuffle(test_pairs)
        
        # ��¼�������ݷֲ�
        logging.info("\nTest data distribution:")
        for min_diff, max_diff, label in perf_ranges:
            count = sum(1 for pair in test_pairs if min_diff <= pair['performance_diff'] < max_diff)
            logging.info(f"  {label} [{min_diff:.2f}, {max_diff if max_diff != float('inf') else '��'}): {count} pairs")
        
        logging.info(f"Final test set size: {len(test_pairs)} pairs")
        
        # ������Զ���Ϣ
        test_pairs_info = {
            'test_pairs': [{
                'code1_id': pair['code1_id'],
                'code2_id': pair['code2_id'],
                'code1_performance': pair['code1_performance'],
                'code2_performance': pair['code2_performance'],
                'is_code1_better': pair['is_code1_better'],
                'performance_diff': pair['performance_diff'],
                'problem_idx': pair['problem_idx']
            } for pair in test_pairs],
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        test_pairs_path = f'./results/test_pairs_{test_pairs_info["timestamp"]}.json'
        with open(test_pairs_path, 'w') as f:
            json.dump(test_pairs_info, f, indent=2)
        
        return train_pairs, test_pairs
    
    def train(self, train_pairs, batch_size=64, epochs=200):
        """ѵ��ģ��"""
        logging.info("Starting training...")
        logging.info(f"Total training pairs: {len(train_pairs)}")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Max epochs: {epochs}")
        
        best_loss = float('inf')
        patience = 15  # ������ͣ����ֵ
        patience_counter = 0
        
        # ���ѧϰ�ʵ�����
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            # ÿ��epoch���´�������
            random.shuffle(train_pairs)
            n_batches = (len(train_pairs) + batch_size - 1) // batch_size
            
            for i in range(0, len(train_pairs), batch_size):
                batch_pairs = train_pairs[i:i+batch_size]
                
                code1_batch = torch.FloatTensor([p['code1_embedding'] for p in batch_pairs]).to(self.device)
                code2_batch = torch.FloatTensor([p['code2_embedding'] for p in batch_pairs]).to(self.device)
                
                targets = torch.FloatTensor([
                    1.0 if p['code1_performance'] < p['code2_performance'] else 0.0
                    for p in batch_pairs
                ]).to(self.device)
                
                self.optimizer.zero_grad()
                scores = self.model(code1_batch, code2_batch).squeeze()
                loss = self.criterion(scores, targets)
                
                # ���L2����
                l2_lambda = 0.001
                l2_reg = torch.tensor(0.).to(self.device)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += l2_lambda * l2_reg
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                predictions = (scores > 0).float()
                correct = (predictions == targets).sum().item()
                correct_predictions += correct
                total_predictions += len(batch_pairs)
            
            avg_loss = total_loss / n_batches
            epoch_accuracy = correct_predictions / total_predictions
            
            # ����ѧϰ�ʵ�����
            scheduler.step(avg_loss)
            
            logging.info(f"Epoch {epoch+1} Results:")
            logging.info(f"  Average Loss: {avg_loss:.4f}")
            logging.info(f"  Training Accuracy: {epoch_accuracy:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # �������ģ��
                self.save_model(f'./results/best_model_epoch_{epoch+1}.pth')
                logging.info("  New best loss achieved! Model saved.")
            else:
                patience_counter += 1
                logging.info(f"  No improvement for {patience_counter} epochs")
            
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        logging.info("\nTraining completed!")
        logging.info(f"Best loss achieved: {best_loss:.4f}")
    
    def evaluate(self, test_pairs):
        """����ģ���ڲ��Լ��ϵı���"""
        logging.info("Starting model evaluation...")
        self.model.eval()
        
        results = {
            'very_poor': {'correct': 0, 'total': 0},
            'poor': {'correct': 0, 'total': 0},
            'significant': {'correct': 0, 'total': 0},
            'moderate': {'correct': 0, 'total': 0},
            'slight': {'correct': 0, 'total': 0},
            'minimal': {'correct': 0, 'total': 0}
        }
        
        # ��̬�����������ֵ�
        problem_results = {}
        for pair in test_pairs:
            prob_idx = pair['problem_idx']
            if prob_idx not in problem_results:
                problem_results[prob_idx] = {'correct': 0, 'total': 0}
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for pair in test_pairs:
                code1_tensor = torch.FloatTensor(pair['code1_embedding']).unsqueeze(0).to(self.device)
                code2_tensor = torch.FloatTensor(pair['code2_embedding']).unsqueeze(0).to(self.device)
                
                diff_score = self.model(code1_tensor, code2_tensor).item()
                predicted_better = diff_score > 0
                
                # ���������ض��Ľ��
                prob_idx = pair['problem_idx']
                problem_results[prob_idx]['total'] += 1
                if predicted_better == pair['is_code1_better']:
                    problem_results[prob_idx]['correct'] += 1
                
                # �������ܲ�����࣬��ѵ��ʱ����һ��
                perf_diff = pair['performance_diff']
                if perf_diff > 0.5:
                    category = 'very_poor'
                elif perf_diff > 0.3:
                    category = 'poor'
                elif perf_diff > 0.2:
                    category = 'significant'
                elif perf_diff > 0.1:
                    category = 'moderate'
                elif perf_diff > 0.05:
                    category = 'slight'
                else:
                    category = 'minimal'
                
                results[category]['total'] += 1
                if predicted_better == pair['is_code1_better']:
                    results[category]['correct'] += 1
                
                predictions.append(predicted_better)
                true_labels.append(pair['is_code1_better'])
        
        # ��Ӱ���������ܱ���
        logging.info("\nPer-Problem Results:")
        for prob_idx, stats in problem_results.items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                logging.info(f"Problem {prob_idx}:")
                logging.info(f"  Total pairs: {stats['total']}")
                logging.info(f"  Correct predictions: {stats['correct']}")
                logging.info(f"  Accuracy: {accuracy:.4f}")
        
        # ��������׼ȷ��
        total_correct = sum(p == t for p, t in zip(predictions, true_labels))
        total_accuracy = total_correct / len(predictions)
        
        # �����ϸ���
        logging.info("\nDetailed Evaluation Results:")
        logging.info("=" * 50)
        
        # �����ܲ���Ӵ�С����������
        category_order = ['very_poor', 'poor', 'significant', 'moderate', 'slight', 'minimal']
        for category in category_order:
            stats = results[category]
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                logging.info(f"{category.upper()}:")
                logging.info(f"  Performance diff: {self.get_diff_range(category)}")
                logging.info(f"  Total pairs: {stats['total']}")
                logging.info(f"  Correct predictions: {stats['correct']}")
                logging.info(f"  Accuracy: {accuracy:.4f}")
        
        logging.info("\nOverall Results:")
        logging.info(f"Total test pairs: {len(predictions)}")
        logging.info(f"Total correct predictions: {total_correct}")
        logging.info(f"Overall accuracy: {total_accuracy:.4f}")
        
        # ����ɸѡ׼ȷ�ʣ�ʶ��ϲ�����������
        poor_code_accuracy = 0
        poor_code_total = results['very_poor']['total'] + results['poor']['total']
        if poor_code_total > 0:
            poor_code_correct = results['very_poor']['correct'] + results['poor']['correct']
            poor_code_accuracy = poor_code_correct / poor_code_total
            logging.info("\nPoor Code Detection Results:")
            logging.info(f"Total poor code pairs: {poor_code_total}")
            logging.info(f"Correctly identified: {poor_code_correct}")
            logging.info(f"Poor code detection accuracy: {poor_code_accuracy:.4f}")

        return total_accuracy, results, predictions, true_labels, poor_code_accuracy

    def get_diff_range(self, category):
        """����ÿ������Ӧ�����ܲ��췶Χ����"""
        ranges = {
            'very_poor': '> 0.5',
            'poor': '0.3 - 0.5',
            'significant': '0.2 - 0.3',
            'moderate': '0.1 - 0.2',
            'slight': '0.05 - 0.1',
            'minimal': '�� 0.05'
        }
        return ranges[category]

    def load_data(self, code_results_path):
        """�������ݼ�"""
        logging.info("Loading data...")
        
        with open(code_results_path, 'r') as f:
            data = json.load(f)
        
        code_embeddings = {}
        code_performances = {}
        
        for item in data['results']:
            code_id = str(item['id'])
            code_embeddings[code_id] = item['embedding']
            code_performances[code_id] = item['results']  # ������������ʵ�������ܽ��
        
        logging.info(f"Loaded {len(code_embeddings)} code samples")
        
        # ������ֱ����ͳ����Ϣ
        num_problems = len(data['results'][0]['results'])
        for prob_idx in range(num_problems):
            problem_perfs = [perfs[prob_idx] for perfs in code_performances.values()]
            logging.info(f"\nProblem {prob_idx} statistics:")
            logging.info(f"  Mean: {np.mean(problem_perfs):.4f}")
            logging.info(f"  Std: {np.std(problem_perfs):.4f}")
            logging.info(f"  Min: {min(problem_perfs):.4f}")
            logging.info(f"  Max: {max(problem_perfs):.4f}")
        
        return code_embeddings, code_performances

    def save_model(self, path):
        """����ģ��"""
        model_info = {
            'state_dict': self.model.state_dict(),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        torch.save(model_info, path)
        logging.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """����ģ��"""
        logging.info(f"Loading model from {path}")
        model_info = torch.load(path, map_location=self.device)
        self.model.load_state_dict(model_info['state_dict'])
        logging.info(f"Model loaded (saved at {model_info['timestamp']})")
    
    def predict_pair(self, code1_embedding, code2_embedding):
        """Ԥ�����ݴ�����������"""
        self.model.eval()
        with torch.no_grad():
            code1_tensor = torch.FloatTensor(code1_embedding).unsqueeze(0).to(self.device)
            code2_tensor = torch.FloatTensor(code2_embedding).unsqueeze(0).to(self.device)
            
            score = self.model(code1_tensor, code2_tensor).item()
            is_code1_better = score > 0
            confidence = abs(score)  # ʹ�÷����ľ���ֵ��Ϊ���Ŷ�
            
            return {
                'is_code1_better': is_code1_better,
                'confidence': confidence,
                'raw_score': score
            }

def main():
    # �������������ȷ�����ظ���
    random.seed(42)
    torch.manual_seed(42)
    
    # ����·��
    code_results_path = 'cvrp_lehd_results_with_embeddings.json'
    model_path = './results/best_performance_predictor.pth'
    
    # �������Ŀ¼
    os.makedirs('./results', exist_ok=True)
    
    # ����Ԥ����
    predictor = CodePerformancePredictor()
    
    # ����Ƿ�Ϊѵ��ģʽ
    is_training = not os.path.exists(model_path) or '--train' in sys.argv
    
    if is_training:
        logging.info("Starting training process...")
        
        # ��������
        code_embeddings, code_performances = predictor.load_data(code_results_path)
        
        # ׼��ѵ�����ݺͲ�������
        train_data, test_pairs = predictor.prepare_training_data(
            code_embeddings, code_performances
        )
        
        # ѵ��ģ��
        predictor.train(train_data)
        
        # ����ģ��
        predictor.save_model(model_path)
        
        # ����ģ��
        accuracy, results, predictions, true_labels, poor_code_accuracy = predictor.evaluate(test_pairs)
        
        # �����������
        results_path = f'./results/evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_path, 'w') as f:
            json.dump({
                'accuracy': accuracy,
                'detailed_results': {
                    category: {
                        'correct': stats['correct'],
                        'total': stats['total'],
                        'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                    }
                    for category, stats in results.items()
                },
                'predictions': predictions,
                'true_labels': true_labels,
                'poor_code_accuracy': poor_code_accuracy
            }, f, indent=2)
        
        logging.info(f"Evaluation results saved to {results_path}")
    
    else:
        logging.info("Loading pre-trained model for prediction...")
        predictor.load_model(model_path)
        
        # ʾ��������һЩ������бȽ�
        with open(code_results_path, 'r') as f:
            data = json.load(f)
            
        # ѡ�����������������бȽ�
        code1 = data['results'][0]
        code2 = data['results'][1]
        
        # ��̬��ȡ��������
        num_problems = len(code1['results'])
        
        # ��ÿ������ֱ����Ԥ��
        for prob_idx in range(num_problems):
            result = predictor.predict_pair(
                code1['embedding'], 
                code2['embedding']
            )
            
            logging.info(f"\nPrediction for Problem {prob_idx}:")
            logging.info(f"Comparing code {code1['id']} and {code2['id']}")
            logging.info(f"Prediction: Code {1 if result['is_code1_better'] else 2} is better")
            logging.info(f"Confidence: {result['confidence']:.4f}")
            logging.info(f"Raw score: {result['raw_score']:.4f}")
            
            # ���ʵ��������Ϊ�ο�
            perf1 = code1['results'][prob_idx]
            perf2 = code2['results'][prob_idx]
            logging.info(f"\nActual performances:")
            logging.info(f"Code 1: {perf1:.4f}")
            logging.info(f"Code 2: {perf2:.4f}")
            logging.info(f"Actually better: Code {1 if perf1 < perf2 else 2}")

if __name__ == "__main__":
    main()
