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
        train_pairs = []
        train_codes = list(train_code_ids)
        
        # �������п��ܵĴ���Բ��������ܲ���
        potential_pairs = []
        for i, code1_id in enumerate(train_codes):
            perf1 = code_performances[code1_id]
            for code2_id in train_codes[i+1:]:
                perf2 = code_performances[code2_id]
                perf_diff = abs(perf1 - perf2)
                
                if perf_diff > 0.05:  # ���ܲ�����ֵ
                    potential_pairs.append({
                        'code1_id': code1_id,
                        'code2_id': code2_id,
                        'code1_embedding': code_embeddings[code1_id],
                        'code2_embedding': code_embeddings[code2_id],
                        'code1_performance': perf1,
                        'code2_performance': perf2,
                        'perf_diff': perf_diff
                    })
        
        # �����ܲ���Խ���ɸѡ�ͷ���
        performance_groups = {
            'significant': [],  # ���ܲ��� > 0.2
            'medium': [],      # 0.1 < ���� <= 0.2
            'small': []        # 0.05 < ���� <= 0.1
        }
        
        for pair in potential_pairs:
            diff = pair['perf_diff']
            if diff > 0.2:
                performance_groups['significant'].append(pair)
            elif diff > 0.1:
                performance_groups['medium'].append(pair)
            elif diff > 0.05:
                performance_groups['small'].append(pair)
        
        # ��ÿ������ѡ��������ȷ������ƽ��
        train_pairs = []
        samples_per_group = min(2000, min(len(g) for g in performance_groups.values()))
        
        for group in performance_groups.values():
            random.shuffle(group)
            train_pairs.extend(group[:samples_per_group])
        
        random.shuffle(train_pairs)
        logging.info(f"Selected {len(train_pairs)} balanced training pairs")
        
        # ׼���������ݶ�
        test_pairs = []
        test_codes = list(test_code_ids)
        
        # Ϊ���Լ��������п��ܵĴ����
        potential_test_pairs = []
        for i, code1_id in enumerate(test_codes):
            perf1 = code_performances[code1_id]
            for code2_id in test_codes[i+1:]:
                perf2 = code_performances[code2_id]
                perf_diff = abs(perf1 - perf2)
                
                potential_test_pairs.append({
                    'code1_id': code1_id,
                    'code2_id': code2_id,
                    'code1_embedding': code_embeddings[code1_id],
                    'code2_embedding': code_embeddings[code2_id],
                    'code1_performance': perf1,
                    'code2_performance': perf2,
                    'is_code1_better': perf1 < perf2,
                    'performance_diff': perf_diff
                })
        
        # ���Լ�����Ҳ���÷ֲ����
        test_groups = {
            'significant': [],
            'medium': [],
            'small': [],
            'very_small': []  # ��Ӹ�С�������
        }
        
        for pair in potential_test_pairs:
            diff = pair['performance_diff']
            if diff > 0.2:
                test_groups['significant'].append(pair)
            elif diff > 0.1:
                test_groups['medium'].append(pair)
            elif diff > 0.05:
                test_groups['small'].append(pair)
            else:
                test_groups['very_small'].append(pair)
        
        # ��ÿ������ѡ���������
        test_pairs = []
        samples_per_test_group = 250  # ÿ��250���������ܹ�1000��
        
        for group in test_groups.values():
            if group:
                selected = random.sample(group, min(samples_per_test_group, len(group)))
                test_pairs.extend(selected)
        
        random.shuffle(test_pairs)
        logging.info(f"Selected {len(test_pairs)} balanced test pairs")
        
        # ������Զ���Ϣ
        test_pairs_info = {
            'test_pairs': [{
                'code1_id': pair['code1_id'],
                'code2_id': pair['code2_id'],
                'code1_performance': pair['code1_performance'],
                'code2_performance': pair['code2_performance'],
                'is_code1_better': pair['is_code1_better'],
                'performance_diff': pair['performance_diff']
            } for pair in test_pairs],
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        test_pairs_path = f'./results/test_pairs_{test_pairs_info["timestamp"]}.json'
        with open(test_pairs_path, 'w') as f:
            json.dump(test_pairs_info, f, indent=2)
        
        return train_pairs, test_pairs
    
    def train(self, train_pairs, batch_size=32, epochs=100):
        """ѵ��ģ��"""
        logging.info("Starting training...")
        logging.info(f"Total training pairs: {len(train_pairs)}")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Max epochs: {epochs}")
        
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            # ��ӡ��ǰepoch��ʼ
            logging.info(f"\nEpoch {epoch+1}/{epochs}")
            logging.info("Processing batches...")
            
            random.shuffle(train_pairs)
            n_batches = (len(train_pairs) + batch_size - 1) // batch_size
            
            for i in range(0, len(train_pairs), batch_size):
                batch_pairs = train_pairs[i:i+batch_size]
                
                code1_batch = torch.FloatTensor([p['code1_embedding'] for p in batch_pairs]).to(self.device)
                code2_batch = torch.FloatTensor([p['code2_embedding'] for p in batch_pairs]).to(self.device)
                
                # ʹ��0/1��ǩ������-1/1
                targets = torch.FloatTensor([
                    1.0 if p['code1_performance'] < p['code2_performance'] else 0.0
                    for p in batch_pairs
                ]).to(self.device)
                
                # ǰ�򴫲�
                scores = self.model(code1_batch, code2_batch).squeeze()
                loss = self.criterion(scores, targets)
                
                # ����׼ȷ��
                predictions = (scores > 0).float()
                correct = (predictions == targets).sum().item()
                correct_predictions += correct
                total_predictions += len(batch_pairs)
                
                # ���򴫲�
                self.optimizer.zero_grad()
                loss.backward()
                
                # �ݶȲü�
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += loss.item()
            
            # ����epochͳ����Ϣ
            avg_loss = total_loss / n_batches
            epoch_accuracy = correct_predictions / total_predictions
            
            # ���epoch���
            logging.info(f"Epoch {epoch+1} Results:")
            logging.info(f"  Average Loss: {avg_loss:.4f}")
            logging.info(f"  Training Accuracy: {epoch_accuracy:.4f}")
            
            # ��ͣ���
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                logging.info("  New best loss achieved!")
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
            'very_close': {'correct': 0, 'total': 0},    # 0-0.05
            'close': {'correct': 0, 'total': 0},         # 0.05-0.1
            'medium': {'correct': 0, 'total': 0},        # 0.1-0.2
            'far': {'correct': 0, 'total': 0}           # >0.2
        }
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for pair in test_pairs:
                code1_tensor = torch.FloatTensor(pair['code1_embedding']).unsqueeze(0).to(self.device)
                code2_tensor = torch.FloatTensor(pair['code2_embedding']).unsqueeze(0).to(self.device)
                
                diff_score = self.model(code1_tensor, code2_tensor).item()
                predicted_better = diff_score > 0
                
                # �������ܲ������
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
        
        # ��������׼ȷ��
        total_correct = sum(p == t for p, t in zip(predictions, true_labels))
        total_accuracy = total_correct / len(predictions)
        
        # �����ϸ���
        logging.info("\nDetailed Evaluation Results:")
        logging.info("=" * 50)
        for category, stats in results.items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                logging.info(f"{category}:")
                logging.info(f"  Total pairs: {stats['total']}")
                logging.info(f"  Correct predictions: {stats['correct']}")
                logging.info(f"  Accuracy: {accuracy:.4f}")
        
        logging.info("\nOverall Results:")
        logging.info(f"Total test pairs: {len(predictions)}")
        logging.info(f"Total correct predictions: {total_correct}")
        logging.info(f"Overall accuracy: {total_accuracy:.4f}")
        
        return total_accuracy, results, predictions, true_labels

    def load_data(self, code_results_path):
        """�������ݼ�"""
        logging.info("Loading data...")
        
        # ���ش�������Ƕ��
        with open(code_results_path, 'r') as f:
            data = json.load(f)
        
        code_embeddings = {}
        code_performances = {}
        
        for item in data['results']:
            code_id = str(item['id'])
            code_embeddings[code_id] = item['embedding']
            # ����ƽ������
            avg_performance = sum(item['results']) / len(item['results'])
            code_performances[code_id] = avg_performance
        
        logging.info(f"Loaded {len(code_embeddings)} code samples")
        
        # �������ͳ����Ϣ
        performances = list(code_performances.values())
        logging.info(f"Performance statistics:")
        logging.info(f"  Mean: {np.mean(performances):.4f}")
        logging.info(f"  Std: {np.std(performances):.4f}")
        logging.info(f"  Min: {min(performances):.4f}")
        logging.info(f"  Max: {max(performances):.4f}")
        
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
    
    # ����·�� - �������·��
    # model_path = './results/best_performance_predictor.pth'
    model_path = './results/model_iter0_finetune1.pth'
    test_pairs_path = './results/test_pairs_20250105_102116.json'
    results_path = '../../tsp_gls_results_with_embeddings.json'  # �޸���һ�У���������Ŀ¼
    
    # �������Ŀ¼
    os.makedirs('./results', exist_ok=True)
    
    # ����Ԥ����������ģ��
    predictor = CodePerformancePredictor()
    predictor.load_model(model_path)
    
    # ���ز��Լ�����
    with open(test_pairs_path, 'r') as f:
        test_pairs_data = json.load(f)
    test_pairs = test_pairs_data['test_pairs'][:100] # ֻȡǰ100�����Զ�
    
    # ���ش���������
    with open(results_path, 'r') as f:
        results_data = json.load(f)
    
    # ����id��embedding��ӳ��
    id_to_embedding = {}
    for item in results_data['results']:
        id_to_embedding[str(item['id'])] = item['embedding']
    
    # �������Զ��б�
    test_data = []
    for pair in test_pairs:
        code1_id = pair['code1_id'] 
        code2_id = pair['code2_id']
        
        # ȷ�����������embedding������
        if code1_id in id_to_embedding and code2_id in id_to_embedding:
            test_data.append({
                'code1_embedding': id_to_embedding[code1_id],
                'code2_embedding': id_to_embedding[code2_id],
                'is_code1_better': pair['is_code1_better'],
                'performance_diff': pair['performance_diff']
            })
    
    # ����ģ��
    logging.info("Starting model evaluation...")
    predictor.model.eval()
    
    correct = 0
    total = len(test_data)
    
    with torch.no_grad():
        for pair in test_data:
            code1_tensor = torch.FloatTensor(pair['code1_embedding']).unsqueeze(0).to(predictor.device)
            code2_tensor = torch.FloatTensor(pair['code2_embedding']).unsqueeze(0).to(predictor.device)
            
            diff_score = predictor.model(code1_tensor, code2_tensor).item()
            predicted_better = diff_score > 0
            
            if predicted_better == pair['is_code1_better']:
                correct += 1
    
    accuracy = correct / total
    logging.info(f"\nEvaluation Results:")
    logging.info(f"Total test pairs: {total}")
    logging.info(f"Correct predictions: {correct}")
    logging.info(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 