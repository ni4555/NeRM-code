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
from typing import List
from pathlib import Path

# 修改日志设置函数
def setup_logging(force_new=False):
    """设置日志记录器
    
    Args:
        force_new (bool): 是否强制创建新的日志记录器，默认False
    """
    logger = logging.getLogger('predictor')
    
    # 如果logger已经有handler且不强制创建新的，直接返回现有logger
    if logger.handlers and not force_new:
        return logger
        
    # 清除现有的handlers
    logger.handlers.clear()
    
    # 设置日志级别
    logger.setLevel(logging.INFO)
    
    # 创建日志文件名
    log_filename = f'predictor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # 创建handlers
    file_handler = logging.FileHandler(log_filename)
    console_handler = logging.StreamHandler()
    
    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger



class PerformancePredictor(nn.Module):
    def __init__(self, code_dim=1536):
        super(PerformancePredictor, self).__init__()
        # Replace the existing logging configuration with a call to setup_logging()
        setup_logging()
        
        # 特征提取器
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
        
        # 比较网络
        self.comparison = nn.Sequential(
            nn.Linear(384, 192),  # 384 = 192 * 2 (两个代码的特征拼接)
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(192, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # 将输出限制在[-1, 1]范围内
        )
    
    def forward(self, x1, x2):
        # 提取特征
        feat1 = self.feature_extractor(x1)
        feat2 = self.feature_extractor(x2)
        
        # 拼接特征并计算相对得分
        combined = torch.cat([feat1, feat2], dim=1)
        return self.comparison(combined)

class CodePerformancePredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PerformancePredictor().to(self.device)
        # 使用BCE损失，将问题转化为二分类
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.01)
        
        # 添加微调相关的属性
        self.training_pairs = []  # 存储训练数据对
        self.min_pairs_for_finetune = 8  # 最小需要多少对数据才进行微调
        self.finetune_batch_size = 4  # 微调时的batch size
        self.finetune_epochs = 5  # 微调时的epoch数
        
        # 创建results目录
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        # 设置历史记录文件路径
        self.history_file = self.results_dir / 'predictor_history.json'
        self.iteration_predictions_file = self.results_dir / 'iteration_predictions.json'
        self.finetune_predictions_file = self.results_dir / 'finetune_predictions.json'
        
        # 添加模型文件记录
        self.model_dir = self.results_dir / 'models'
        self.model_dir.mkdir(exist_ok=True)
        
        # 修改预测历史结构
        self.prediction_history = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'finetune_rounds': 0,
            'round_accuracies': [],
            'predictions': [],
            'iterations': [],
            'current_iteration': {
                'iteration_id': 0,
                'start_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'code_pairs': [],
                'predictions': [],
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0.0,
                'model_files': []  # 记录本轮迭代使用的模型文件
            }
        }
        
        # 修改迭代预测记录结构
        self.iteration_predictions = {
            'iterations': []
        }
        
        # 修改微调预测记录结构
        self.finetune_predictions = {
            'finetune_rounds': []
        }
        
        # 如果历史记录文件存在，加载它
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self.prediction_history = json.load(f)
                logging.info(f"Loaded existing prediction history from {self.history_file}")
            except Exception as e:
                logging.error(f"Error loading prediction history: {e}")
        
        # 确保至少有一个迭代记录
        if not self.iteration_predictions['iterations']:
            self.iteration_predictions['iterations'].append({
                'iteration_id': 0,
                'start_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'code_pairs': [],
                'predictions': []
            })

    def start_new_iteration(self):
        """开始新的迭代"""
        # 保存上一轮迭代的统计信息
        if self.prediction_history['current_iteration']['total_predictions'] > 0:
            self.prediction_history['current_iteration']['accuracy'] = (
                self.prediction_history['current_iteration']['correct_predictions'] / 
                self.prediction_history['current_iteration']['total_predictions']
            )
            self.prediction_history['iterations'].append(self.prediction_history['current_iteration'])
            
            # 保存迭代预测记录
            self._save_iteration_predictions()
        
        # 初始化新一轮迭代
        self.prediction_history['current_iteration'] = {
            'iteration_id': len(self.prediction_history['iterations']),
            'start_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'code_pairs': [],
            'predictions': [],
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0
        }
        
        # 初始化新一轮迭代的预测记录
        self.iteration_predictions['iterations'].append({
            'iteration_id': self.prediction_history['current_iteration']['iteration_id'],
            'start_time': self.prediction_history['current_iteration']['start_time'],
            'code_pairs': [],
            'predictions': []
        })
        
        # 保存更新后的历史记录
        self._save_prediction_history()

    def _save_iteration_predictions(self):
        """保存迭代预测记录"""
        try:
            with open(self.iteration_predictions_file, 'w', encoding='utf-8') as f:
                json.dump(self.iteration_predictions, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved iteration predictions to {self.iteration_predictions_file}")
        except Exception as e:
            logging.error(f"Error saving iteration predictions: {e}")

    def _save_finetune_predictions(self):
        """保存微调预测记录"""
        try:
            with open(self.finetune_predictions_file, 'w', encoding='utf-8') as f:
                json.dump(self.finetune_predictions, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved finetune predictions to {self.finetune_predictions_file}")
        except Exception as e:
            logging.error(f"Error saving finetune predictions: {e}")

    def add_training_pair(self, code1_embedding, code2_embedding, code1_performance, code2_performance, code1_content=None, code2_content=None):
        """添加一个训练数据对并更新预测历史"""
        # 确保迭代记录存在
        if not self.iteration_predictions['iterations']:
            self.iteration_predictions['iterations'].append({
                'iteration_id': len(self.prediction_history['iterations']),
                'start_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'code_pairs': [],
                'predictions': []
            })
        
        # 保存训练数据，包括代码内容
        self.training_pairs.append({
            'code1_embedding': code1_embedding,
            'code2_embedding': code2_embedding,
            'code1_performance': code1_performance,
            'code2_performance': code2_performance,
            'code1_content': code1_content,
            'code2_content': code2_content
        })
        
        # 更新预测历史中的正确预测计数
        if code1_performance < code2_performance:
            self.prediction_history['correct_predictions'] += 1
            self.prediction_history['current_iteration']['correct_predictions'] += 1
        
        # 更新当前迭代的代码对记录，包含代码内容
        code_pair_record = {
            'code1_performance': code1_performance,
            'code2_performance': code2_performance,
            'code1_content': code1_content,
            'code2_content': code2_content,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        self.prediction_history['current_iteration']['code_pairs'].append(code_pair_record)
        
        # 更新迭代预测记录，包含代码内容
        current_iteration = self.iteration_predictions['iterations'][-1]
        current_iteration['code_pairs'].append(code_pair_record)
        
        # 如果积累了足够的数据对，进行微调
        if len(self.training_pairs) >= self.min_pairs_for_finetune:
            self.finetune()
            self.training_pairs = []  # 清空训练数据对
    
    def finetune(self):
        """使用积累的训练数据对模型进行微调"""
        if len(self.training_pairs) < self.min_pairs_for_finetune:
            return
            
        logging.info(f"Starting finetuning with {len(self.training_pairs)} training pairs")
        
        # 准备训练数据
        train_data = []
        for pair in self.training_pairs:
            label = 1.0 if pair['code1_performance'] < pair['code2_performance'] else 0.0
            train_data.append({
                'code1': torch.FloatTensor(pair['code1_embedding']).to(self.device),
                'code2': torch.FloatTensor(pair['code2_embedding']).to(self.device),
                'label': torch.FloatTensor([label]).to(self.device)
            })
        
        # 进行微调
        self.model.train()
        for epoch in range(self.finetune_epochs):
            total_loss = 0
            n_batches = (len(train_data) + self.finetune_batch_size - 1) // self.finetune_batch_size
            
            for i in range(0, len(train_data), self.finetune_batch_size):
                batch = train_data[i:i + self.finetune_batch_size]
                
                code1_batch = torch.stack([item['code1'] for item in batch])
                code2_batch = torch.stack([item['code2'] for item in batch])
                labels = torch.cat([item['label'] for item in batch])
                
                scores = self.model(code1_batch, code2_batch).squeeze()
                loss = self.criterion(scores, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches
            logging.info(f"Finetuning epoch {epoch+1}/{self.finetune_epochs}, avg loss: {avg_loss:.4f}")
        
        # 计算微调后的预测准确率
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for pair in train_data:
                code1 = pair['code1'].unsqueeze(0)
                code2 = pair['code2'].unsqueeze(0)
                label = pair['label']
                
                score = self.model(code1, code2).item()
                prediction = 1.0 if score > 0 else 0.0
                
                if prediction == label:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        self.prediction_history['finetune_rounds'] += 1
        self.prediction_history['round_accuracies'].append(accuracy)
        
        # 保存微调后的模型
        model_filename = f'model_iter{self.prediction_history["current_iteration"]["iteration_id"]}_finetune{self.prediction_history["finetune_rounds"]}.pth'
        model_path = self.model_dir / model_filename
        self.save_model(model_path)
        
        # 记录当前迭代使用的模型文件
        self.prediction_history['current_iteration']['model_files'].append({
            'filename': model_filename,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'accuracy': accuracy,
            'loss': avg_loss
        })
        
        # 记录微调轮次的预测结果
        self.finetune_predictions['finetune_rounds'].append({
            'round_id': self.prediction_history['finetune_rounds'],
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'training_pairs': len(self.training_pairs),
            'accuracy': accuracy,
            'loss': avg_loss,
            'model_file': model_filename,
            'training_data': [
                {
                    'code1_performance': pair['code1_performance'],
                    'code2_performance': pair['code2_performance'],
                    'code1_content': pair['code1_content'],
                    'code2_content': pair['code2_content']
                }
                for pair in self.training_pairs
            ]
        })
        
        # 记录预测统计信息
        logging.info(f"Finetuning round {self.prediction_history['finetune_rounds']} completed")
        logging.info(f"Training accuracy: {accuracy:.4f}")
        logging.info(f"Average accuracy across all rounds: {sum(self.prediction_history['round_accuracies']) / len(self.prediction_history['round_accuracies']):.4f}")
        
        # 保存所有历史记录
        self._save_prediction_history()
        self._save_finetune_predictions()
        
        logging.info("Finetuning completed")
    
    def _save_prediction_history(self):
        """保存预测历史到文件"""
        try:
            # 计算总体准确率
            if self.prediction_history['total_predictions'] > 0:
                self.prediction_history['accuracy'] = self.prediction_history['correct_predictions'] / self.prediction_history['total_predictions']
            
            # 计算当前迭代的准确率
            if self.prediction_history['current_iteration']['total_predictions'] > 0:
                self.prediction_history['current_iteration']['accuracy'] = (
                    self.prediction_history['current_iteration']['correct_predictions'] / 
                    self.prediction_history['current_iteration']['total_predictions']
                )
            
            # 保存到文件
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.prediction_history, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved prediction history to {self.history_file}")
            logging.info(f"Current iteration accuracy: {self.prediction_history['current_iteration']['accuracy']:.4f}")
            logging.info(f"Overall accuracy: {self.prediction_history['accuracy']:.4f}")
        except Exception as e:
            logging.error(f"Error saving prediction history: {e}")
            logging.error(f"Current directory: {os.getcwd()}")
            logging.error(f"History file path: {self.history_file}")
    
    def prepare_training_data(self, code_embeddings, code_performances, test_size=0.2):
        """准备训练数据和测试数据"""
        logging.info("Preparing training and test data...")
        
        # 随机分割代码ID为训练集和测试集
        code_ids = list(code_embeddings.keys())
        n_test = int(len(code_ids) * test_size)
        test_code_ids = set(random.sample(code_ids, n_test))
        train_code_ids = set(code_ids) - test_code_ids
        
        # 保存数据集划分
        split_info = {
            'train_code_ids': list(train_code_ids),
            'test_code_ids': list(test_code_ids),
            'split_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        split_path = f'./results/dataset_split_{split_info["split_timestamp"]}.json'
        with open(split_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        # 生成训练对
        train_pairs = []
        train_codes = list(train_code_ids)
        
        # 生成所有可能的代码对并计算性能差异
        potential_pairs = []
        for i, code1_id in enumerate(train_codes):
            perf1 = code_performances[code1_id]
            for code2_id in train_codes[i+1:]:
                perf2 = code_performances[code2_id]
                perf_diff = abs(perf1 - perf2)
                
                if perf_diff > 0.05:  # 性能差异阈值
                    potential_pairs.append({
                        'code1_id': code1_id,
                        'code2_id': code2_id,
                        'code1_performance': perf1,
                        'code2_performance': perf2,
                        'perf_diff': perf_diff
                    })
        
        # 按性能差异对进行筛选和分组
        performance_groups = {
            'significant': [],  # 性能差异 > 0.2
            'medium': [],      # 0.1 < 差异 <= 0.2
            'small': []        # 0.05 < 差异 <= 0.1
        }
        
        for pair in potential_pairs:
            diff = pair['perf_diff']
            if diff > 0.2:
                performance_groups['significant'].append(pair)
            elif diff > 0.1:
                performance_groups['medium'].append(pair)
            elif diff > 0.05:
                performance_groups['small'].append(pair)
        
        # 从每个组中选择样本，确保数据平衡
        train_pairs = []
        samples_per_group = min(2000, min(len(g) for g in performance_groups.values()))
        
        for group in performance_groups.values():
            random.shuffle(group)
            train_pairs.extend(group[:samples_per_group])
        
        random.shuffle(train_pairs)
        logging.info(f"Selected {len(train_pairs)} balanced training pairs")
        
        # 准备测试数据对
        test_pairs = []
        test_codes = list(test_code_ids)
        
        # 为测试集生成所有可能的代码对
        potential_test_pairs = []
        for i, code1_id in enumerate(test_codes):
            perf1 = code_performances[code1_id]
            for code2_id in test_codes[i+1:]:
                perf2 = code_performances[code2_id]
                perf_diff = abs(perf1 - perf2)
                
                potential_test_pairs.append({
                    'code1_id': code1_id,
                    'code2_id': code2_id,
                    'code1_performance': perf1,
                    'code2_performance': perf2,
                    'is_code1_better': perf1 < perf2,
                    'performance_diff': perf_diff
                })
        
        # 测试集构建也采用分层抽样
        test_groups = {
            'significant': [],
            'medium': [],
            'small': [],
            'very_small': []  # 添加更小差异的组
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
        
        # 从每个组中选择测试样本
        test_pairs = []
        samples_per_test_group = 250  # 每组250个样本，总共1000个
        
        for group in test_groups.values():
            if group:
                selected = random.sample(group, min(samples_per_test_group, len(group)))
                test_pairs.extend(selected)
        
        random.shuffle(test_pairs)
        logging.info(f"Selected {len(test_pairs)} balanced test pairs")
        
        # 保存测试对信息
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
        """训练模型"""
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
            
            # 打印当前epoch开始
            logging.info(f"\nEpoch {epoch+1}/{epochs}")
            logging.info("Processing batches...")
            
            random.shuffle(train_pairs)
            n_batches = (len(train_pairs) + batch_size - 1) // batch_size
            
            for i in range(0, len(train_pairs), batch_size):
                batch_pairs = train_pairs[i:i+batch_size]
                
                code1_batch = torch.FloatTensor([p['code1_embedding'] for p in batch_pairs]).to(self.device)
                code2_batch = torch.FloatTensor([p['code2_embedding'] for p in batch_pairs]).to(self.device)
                
                # 使用0/1标签而不是-1/1
                targets = torch.FloatTensor([
                    1.0 if p['code1_performance'] < p['code2_performance'] else 0.0
                    for p in batch_pairs
                ]).to(self.device)
                
                # 前向传播
                scores = self.model(code1_batch, code2_batch).squeeze()
                loss = self.criterion(scores, targets)
                
                # 计算准确率
                predictions = (scores > 0).float()
                correct = (predictions == targets).sum().item()
                correct_predictions += correct
                total_predictions += len(batch_pairs)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += loss.item()
            
            # 计算epoch统计信息
            avg_loss = total_loss / n_batches
            epoch_accuracy = correct_predictions / total_predictions
            
            # 输出epoch结果
            logging.info(f"Epoch {epoch+1} Results:")
            logging.info(f"  Average Loss: {avg_loss:.4f}")
            logging.info(f"  Training Accuracy: {epoch_accuracy:.4f}")
            
            # 早停检查
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
        """评估模型在测试集上的表现"""
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
        
        # 计算总体准确率
        total_correct = sum(p == t for p, t in zip(predictions, true_labels))
        total_accuracy = total_correct / len(predictions)
        
        # 输出详细结果
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
        """加载数据集"""
        logging.info("Loading data...")
        
        # 加载代码结果和嵌入
        with open(code_results_path, 'r') as f:
            data = json.load(f)
        
        code_embeddings = {}
        code_performances = {}
        
        for item in data['results']:
            code_id = str(item['id'])
            code_embeddings[code_id] = item['embedding']
            # 计算平均性能
            avg_performance = sum(item['results']) / len(item['results'])
            code_performances[code_id] = avg_performance
        
        logging.info(f"Loaded {len(code_embeddings)} code samples")
        
        # 输出性能统计信息
        performances = list(code_performances.values())
        logging.info(f"Performance statistics:")
        logging.info(f"  Mean: {np.mean(performances):.4f}")
        logging.info(f"  Std: {np.std(performances):.4f}")
        logging.info(f"  Min: {min(performances):.4f}")
        logging.info(f"  Max: {max(performances):.4f}")
        
        return code_embeddings, code_performances

    def save_model(self, path):
        """保存模型"""
        model_info = {
            'state_dict': self.model.state_dict(),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        torch.save(model_info, path)
        logging.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """加载模型"""
        logging.info(f"Loading model from {path}")
        model_info = torch.load(path, map_location=self.device)
        self.model.load_state_dict(model_info['state_dict'])
        logging.info(f"Model loaded (saved at {model_info['timestamp']})")
    
    def predict_pair(self, code1_embedding, code2_embedding, code1_content=None, code2_content=None):
        """预测两份代码的相对性能"""
        logger = logging.getLogger('predictor')
        logger.info("Starting performance prediction for code pair...")
        
        self.model.eval()
        with torch.no_grad():
            code1_tensor = torch.FloatTensor(code1_embedding).unsqueeze(0).to(self.device)
            code2_tensor = torch.FloatTensor(code2_embedding).unsqueeze(0).to(self.device)      
  
            score = self.model(code1_tensor, code2_tensor).item()
            is_code1_better = score > 0
            confidence = abs(score)
            
            # 更新预测统计
            self.prediction_history['total_predictions'] += 1
            self.prediction_history['current_iteration']['total_predictions'] += 1
            
            # 记录预测结果
            logger.info(f"Prediction complete:")
            logger.info(f"  Raw score: {score:.4f}")
            logger.info(f"  Confidence: {confidence:.4f}")
            logger.info(f"  Prediction: Code {'1' if is_code1_better else '2'} is predicted to perform better")
            
            # 保存预测结果到历史记录，包含代码内容
            prediction_record = {
                'score': score,
                'is_code1_better': is_code1_better,
                'confidence': confidence,
                'code1_content': code1_content,
                'code2_content': code2_content,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            self.prediction_history['predictions'].append(prediction_record)
            self.prediction_history['current_iteration']['predictions'].append(prediction_record)
            
            # 更新迭代预测记录，包含代码内容
            current_iteration = self.iteration_predictions['iterations'][-1]
            current_iteration['predictions'].append(prediction_record)
            
            # 每100次预测保存一次历史记录
            if self.prediction_history['total_predictions'] % 100 == 0:
                self._save_prediction_history()
                self._save_iteration_predictions()
            
            return {
                'is_code1_better': is_code1_better,
                'confidence': confidence,
                'raw_score': score
            }

def main():
    # 设置随机种子以确保可重复性
    random.seed(42)
    torch.manual_seed(42)
    
    # 设置路径
    code_results_path = 'tsp_gls_results_with_embeddings.json'
    # model_path = './results/best_performance_predictor.pth'
    model_path = './results/model_iter0_finetune1.pth'
    
    # 创建结果目录
    os.makedirs('./results', exist_ok=True)
    
    # 创建预测器
    predictor = CodePerformancePredictor()
    
    # 检查是否为训练模式
    is_training = not os.path.exists(model_path) or '--train' in sys.argv
    
    if is_training:
        logging.info("Starting training process...")
        
        # 加载数据
        code_embeddings, code_performances = predictor.load_data(code_results_path)
        
        # 准备训练数据和测试数据
        train_data, test_pairs = predictor.prepare_training_data(
            code_embeddings, code_performances
        )
        
        # 训练模型
        predictor.train(train_data)
        
        # 保存模型
        predictor.save_model(model_path)
        
        # 评估模型
        accuracy, results, predictions, true_labels = predictor.evaluate(test_pairs)
        
        # 保存评估结果
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
                'true_labels': true_labels
            }, f, indent=2)
        
        logging.info(f"Evaluation results saved to {results_path}")
    
    else:
        logging.info("Loading pre-trained model for prediction...")
        predictor.load_model(model_path)
        
        # 这里可以添加预测示例
        logging.info("Model ready for predictions.")
        
        # 示例：加载一些代码进行比较
        with open(code_results_path, 'r') as f:
            data = json.load(f)
            
        # 选择两个代码样本进行比较
        code1 = data['results'][0]
        code2 = data['results'][1]
        
        result = predictor.predict_pair(code1['embedding'], code2['embedding'])
        
        logging.info("\nPrediction Example:")
        logging.info(f"Comparing code {code1['id']} and {code2['id']}")
        logging.info(f"Prediction: Code {1 if result['is_code1_better'] else 2} is better")
        logging.info(f"Confidence: {result['confidence']:.4f}")
        logging.info(f"Raw score: {result['raw_score']:.4f}")
        
        # 输出实际性能作为参考
        perf1 = sum(code1['results']) / len(code1['results'])
        perf2 = sum(code2['results']) / len(code2['results'])
        logging.info(f"\nActual performances:")
        logging.info(f"Code 1: {perf1:.4f}")
        logging.info(f"Code 2: {perf2:.4f}")
        logging.info(f"Actually better: Code {1 if perf1 < perf2 else 2}")

if __name__ == "__main__":
    main() 