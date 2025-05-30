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

# 配置日志
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
        train_codes = list(train_code_ids)
        
        # 生成所有可能的代码对并计算平均性能差异
        potential_pairs = []
        for i, code1_id in enumerate(train_codes):
            perfs1 = code_performances[code1_id]
            # 计算code1的平均性能
            valid_perfs1 = [p for p in perfs1 if p < 1000000 and not np.isinf(p)]
            if not valid_perfs1:  # 如果没有有效性能值，跳过这个代码
                continue
            avg_perf1 = np.mean(valid_perfs1)
            
            for code2_id in train_codes[i+1:]:
                perfs2 = code_performances[code2_id]
                # 计算code2的平均性能
                valid_perfs2 = [p for p in perfs2 if p < 1000000 and not np.isinf(p)]
                if not valid_perfs2:  # 如果没有有效性能值，跳过这个代码
                    continue
                avg_perf2 = np.mean(valid_perfs2)
                
                # 计算相对性能差异 (基于平均性能)
                perf_diff = abs((avg_perf1 - avg_perf2) / max(avg_perf1, avg_perf2))
                
                # 只有当性能有明显差异时才纳入考虑
                if perf_diff > 0.01:
                    potential_pairs.append({
                        'code1_id': code1_id,
                        'code2_id': code2_id,
                        'code1_embedding': code_embeddings[code1_id],
                        'code2_embedding': code_embeddings[code2_id],
                        'code1_performance': avg_perf1,
                        'code2_performance': avg_perf2,
                        'perf_diff': perf_diff
                    })
        
        # 按性能差异排序
        potential_pairs.sort(key=lambda x: x['perf_diff'])
        
        # 修改性能差异的分界点，增加对极端差异的细分
        thresholds = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, float('inf')]  # 增加了更多的高差异区间
        bucket_pairs = [[] for _ in range(len(thresholds)-1)]
        
        # 将数据分配到不同的桶中
        for pair in potential_pairs:
            for i, (low, high) in enumerate(zip(thresholds[:-1], thresholds[1:])):
                if low <= pair['perf_diff'] < high:
                    bucket_pairs[i].append(pair)
                    break
        
        # 计算每个桶的原始样本数量
        original_counts = [len(bucket) for bucket in bucket_pairs]
        total_samples = sum(original_counts)
        
        # 调整目标分布比例，增加对高差异样本的权重
        target_ratios = [0.1, 0.1, 0.15, 0.15, 0.15, 0.1, 0.1, 0.08, 0.07]  # 更细致的差异划分
        
        # 增加基础采样数以确保足够的样本
        base_samples = 4000
        min_samples_per_bucket = 250  # 增加最小样本数
        
        train_pairs = []
        for i, (bucket, target_ratio) in enumerate(zip(bucket_pairs, target_ratios)):
            if bucket:
                # 计算目标样本数
                target_size = max(
                    min_samples_per_bucket,
                    int(base_samples * target_ratio)
                )
                
                # 对于高差异的情况（最后三个桶），保留所有样本并进行过采样
                if i >= len(bucket_pairs) - 3:
                    # 过采样：重复添加高差异样本
                    multiplier = 2 if i == len(bucket_pairs) - 1 else 1.5
                    samples = bucket * int(multiplier)
                    target_size = len(samples)
                    bucket = samples
                
                # 对于中等到大差异的桶，适当增加采样
                elif i in [4, 5, 6]:  # 0.2-0.5 范围
                    target_size = min(int(target_size * 1.3), len(bucket))
                
                # 确保不超过桶中实际样本数
                actual_size = min(target_size, len(bucket))
                
                # 采样
                sampled = random.sample(bucket, actual_size)
                train_pairs.extend(sampled)
                
                logging.info(f"Bucket {i} (diff range: {thresholds[i]:.3f}-{thresholds[i+1] if thresholds[i+1] != float('inf') else '∞'}):")
                logging.info(f"  Original samples: {len(bucket)}")
                logging.info(f"  Sampled: {actual_size}")
                logging.info(f"  Target ratio: {target_ratio:.2f}")
                logging.info(f"  Actual ratio: {actual_size/base_samples:.2f}")
        
        random.shuffle(train_pairs)
        
        # 更新性能范围的记录
        perf_ranges = [
            (0.01, 0.03, "Minimal diff"),
            (0.03, 0.05, "Very small diff"),
            (0.05, 0.1, "Small diff"),
            (0.1, 0.2, "Medium diff"),
            (0.2, 0.3, "Large diff"),
            (0.3, 0.5, "Very large diff"),
            (0.5, float('inf'), "Extreme diff")
        ]
        
        # 记录训练数据分布
        logging.info("\nTraining data distribution:")
        for min_diff, max_diff, label in perf_ranges:
            count = sum(1 for pair in train_pairs if min_diff <= pair['perf_diff'] < max_diff)
            total = sum(1 for pair in potential_pairs if min_diff <= pair['perf_diff'] < max_diff)
            logging.info(f"  {label} [{min_diff:.2f}, {max_diff if max_diff != float('inf') else '∞'}): "
                        f"{count} pairs (from {total} total)")
        
        logging.info(f"Final training set size: {len(train_pairs)} pairs")
        
        # 准备测试数据
        test_pairs = []
        test_codes = list(test_code_ids)
        
        # 为测试集生成代码对
        for i, code1_id in enumerate(test_codes):
            perfs1 = code_performances[code1_id]
            valid_perfs1 = [p for p in perfs1 if p < 1000000 and not np.isinf(p)]
            if not valid_perfs1:
                continue
            avg_perf1 = np.mean(valid_perfs1)
            
            for code2_id in test_codes[i+1:]:
                perfs2 = code_performances[code2_id]
                valid_perfs2 = [p for p in perfs2 if p < 1000000 and not np.isinf(p)]
                if not valid_perfs2:
                    continue
                avg_perf2 = np.mean(valid_perfs2)
                
                # 计算相对性能差异
                perf_diff = abs((avg_perf1 - avg_perf2) / max(avg_perf1, avg_perf2))
                
                # 只有当性能有明显差异时才纳入考虑
                if perf_diff > 0.01:
                    test_pairs.append({
                        'code1_id': code1_id,
                        'code2_id': code2_id,
                        'code1_embedding': code_embeddings[code1_id],
                        'code2_embedding': code_embeddings[code2_id],
                        'code1_performance': avg_perf1,
                        'code2_performance': avg_perf2,
                        'is_code1_better': avg_perf1 > avg_perf2,
                        'performance_diff': perf_diff
                    })
        
        # 修改测试集采样策略，确保各区间样本充足
        samples_per_bucket = [300, 400, 500, 600, 700, 800, 1000]  # 增加大差异区间的测试样本数
        
        test_pairs.sort(key=lambda x: x['performance_diff'])
        sampled_test_pairs = []
        
        # 为测试集使用相似的分桶策略
        test_buckets = [[] for _ in range(len(thresholds)-1)]
        for pair in test_pairs:
            for i, (low, high) in enumerate(zip(thresholds[:-1], thresholds[1:])):
                if low <= pair['performance_diff'] < high:
                    test_buckets[i].append(pair)
                    break
        
        # 从每个桶中采样
        for bucket, target_size in zip(test_buckets, samples_per_bucket):
            if bucket:
                actual_size = min(target_size, len(bucket))
                sampled_test_pairs.extend(random.sample(bucket, actual_size))
        
        test_pairs = sampled_test_pairs
        
        random.shuffle(test_pairs)
        
        # 记录测试数据分布
        logging.info("\nTest data distribution:")
        for min_diff, max_diff, label in perf_ranges:
            count = sum(1 for pair in test_pairs if min_diff <= pair['performance_diff'] < max_diff)
            logging.info(f"  {label} [{min_diff:.2f}, {max_diff if max_diff != float('inf') else '∞'}): {count} pairs")
        
        logging.info(f"Final test set size: {len(test_pairs)} pairs")
        
        # 保存测试对信息
        test_pairs_info = {
            'test_pairs': [{
                'code1_id': pair['code1_id'],
                'code2_id': pair['code2_id'],
                'code1_performance': float(pair['code1_performance']),  # 确保是float类型
                'code2_performance': float(pair['code2_performance']),  # 确保是float类型
                'is_code1_better': int(pair['is_code1_better']),  # 将bool转换为int
                'performance_diff': float(pair['performance_diff'])  # 确保是float类型
            } for pair in test_pairs],
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        test_pairs_path = f'./results/test_pairs_{test_pairs_info["timestamp"]}.json'
        with open(test_pairs_path, 'w') as f:
            json.dump(test_pairs_info, f, indent=2)
        
        return train_pairs, test_pairs
    
    def train(self, train_pairs, batch_size=128, epochs=200):
        """训练模型"""
        logging.info("Starting training...")
        logging.info(f"Total training pairs: {len(train_pairs)}")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Max epochs: {epochs}")
        
        best_loss = float('inf')
        patience = 20  # 增加早停耐心值
        min_lr = 1e-6  # 添加最小学习率阈值
        patience_counter = 0
        
        # 修改学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=7,  # 增加学习率调整耐心值
            verbose=True,
            min_lr=min_lr
        )
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            # 每个epoch重新打乱数据
            random.shuffle(train_pairs)
            n_batches = (len(train_pairs) + batch_size - 1) // batch_size
            
            for i in range(0, len(train_pairs), batch_size):
                batch_pairs = train_pairs[i:i+batch_size]
                
                code1_batch = torch.FloatTensor([p['code1_embedding'] for p in batch_pairs]).to(self.device)
                code2_batch = torch.FloatTensor([p['code2_embedding'] for p in batch_pairs]).to(self.device)
                
                targets = torch.FloatTensor([
                    1.0 if p['code1_performance'] > p['code2_performance'] else 0.0
                    for p in batch_pairs
                ]).to(self.device)
                
                # 获取性能差异用于加权
                performance_diffs = torch.FloatTensor([p['perf_diff'] for p in batch_pairs]).to(self.device)
                
                self.optimizer.zero_grad()
                scores = self.model(code1_batch, code2_batch).squeeze()
                
                # 使用加权损失
                loss = self.weighted_bce_loss(scores, targets, performance_diffs)
                
                # 添加L2正则化
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
            
            # 更新学习率调度器
            scheduler.step(avg_loss)
            
            # 检查当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr <= min_lr:
                logging.info(f"Learning rate {current_lr} has reached minimum threshold. Stopping training.")
                break
            
            logging.info(f"Epoch {epoch+1} Results:")
            logging.info(f"  Average Loss: {avg_loss:.4f}")
            logging.info(f"  Training Accuracy: {epoch_accuracy:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # 保存最佳模型
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
        """评估模型在测试集上的表现"""
        logging.info("Starting model evaluation...")
        self.model.eval()
        
        results = {
            'extreme': {'correct': 0, 'total': 0},
            'very_large': {'correct': 0, 'total': 0},
            'large': {'correct': 0, 'total': 0},
            'medium': {'correct': 0, 'total': 0},
            'small': {'correct': 0, 'total': 0},
            'very_small': {'correct': 0, 'total': 0},
            'minimal': {'correct': 0, 'total': 0}
        }
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for pair in test_pairs:
                code1_tensor = torch.FloatTensor(pair['code1_embedding']).unsqueeze(0).to(self.device)
                code2_tensor = torch.FloatTensor(pair['code2_embedding']).unsqueeze(0).to(self.device)
                
                diff_score = self.model(code1_tensor, code2_tensor).item()
                predicted_better = diff_score > 0
                
                # 根据性能差异分类，与训练时保持一致
                perf_diff = pair['performance_diff']
                if perf_diff > 0.5:
                    category = 'extreme'
                elif perf_diff > 0.3:
                    category = 'very_large'
                elif perf_diff > 0.2:
                    category = 'large'
                elif perf_diff > 0.1:
                    category = 'medium'
                elif perf_diff > 0.05:
                    category = 'small'
                elif perf_diff > 0.03:
                    category = 'very_small'
                else:
                    category = 'minimal'
                
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
        
        # 按性能差异从大到小排序输出结果
        category_order = ['extreme', 'very_large', 'large', 'medium', 'small', 'very_small', 'minimal']
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
        
        # 计算筛选准确率（识别较差代码的能力）
        poor_code_accuracy = 0
        poor_code_total = results['extreme']['total'] + results['very_large']['total'] + results['large']['total']
        if poor_code_total > 0:
            poor_code_correct = results['extreme']['correct'] + results['very_large']['correct'] + results['large']['correct']
            poor_code_accuracy = poor_code_correct / poor_code_total
            logging.info("\nPoor Code Detection Results:")
            logging.info(f"Total poor code pairs: {poor_code_total}")
            logging.info(f"Correctly identified: {poor_code_correct}")
            logging.info(f"Poor code detection accuracy: {poor_code_accuracy:.4f}")

        return total_accuracy, results, predictions, true_labels, poor_code_accuracy

    def get_diff_range(self, category):
        """返回每个类别对应的性能差异范围描述"""
        ranges = {
            'extreme': '> 0.5',
            'very_large': '0.3 - 0.5',
            'large': '0.2 - 0.3',
            'medium': '0.1 - 0.2',
            'small': '0.05 - 0.1',
            'very_small': '0.03 - 0.05',
            'minimal': '≤ 0.03'
        }
        return ranges[category]

    def load_data(self, code_results_path):
        """加载数据集"""
        logging.info("Loading data...")
        
        with open(code_results_path, 'r') as f:
            data = json.load(f)
        
        code_embeddings = {}
        code_performances = {}
        
        for item in data['results']:
            code_id = str(item['id'])
            code_embeddings[code_id] = item['embedding']
            code_performances[code_id] = item['results']  # 保存所有问题实例的性能结果
        
        logging.info(f"Loaded {len(code_embeddings)} code samples")
        
        # 按问题分别计算统计信息
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
    
    def predict_pair(self, code1_embedding, code2_embedding):
        """预测两份代码的相对性能"""
        self.model.eval()
        with torch.no_grad():
            code1_tensor = torch.FloatTensor(code1_embedding).unsqueeze(0).to(self.device)
            code2_tensor = torch.FloatTensor(code2_embedding).unsqueeze(0).to(self.device)
            
            score = self.model(code1_tensor, code2_tensor).item()
            is_code1_better = score > 0
            confidence = abs(score)  # 使用分数的绝对值作为置信度
            
            return {
                'is_code1_better': is_code1_better,
                'confidence': confidence,
                'raw_score': score
            }

    def weighted_bce_loss(self, predictions, targets, performance_diffs):
        """
        加权二元交叉熵损失，对高性能差异样本赋予更高权重
        """
        base_weights = torch.ones_like(targets)
        
        # 根据性能差异调整权重
        for i, diff in enumerate(performance_diffs):
            if diff > 1.0:
                base_weights[i] = 4.0  # 极端差异
            elif diff > 0.8:
                base_weights[i] = 3.5
            elif diff > 0.5:
                base_weights[i] = 3.0
            elif diff > 0.3:
                base_weights[i] = 2.5
            elif diff > 0.2:
                base_weights[i] = 2.0
        
        bce_loss = nn.BCEWithLogitsLoss(weight=base_weights)
        return bce_loss(predictions, targets)

def main():
    # 设置随机种子以确保可重复性
    random.seed(42)
    torch.manual_seed(42)
    
    # 设置路径
    code_results_path = 'mkp_aco_results_with_embeddings.json'
    model_path = './results/best_performance_predictor.pth'
    
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
        accuracy, results, predictions, true_labels, poor_code_accuracy = predictor.evaluate(test_pairs)
        
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
                'true_labels': true_labels,
                'poor_code_accuracy': poor_code_accuracy
            }, f, indent=2)
        
        logging.info(f"Evaluation results saved to {results_path}")
    
    else:
        logging.info("Loading pre-trained model for prediction...")
        predictor.load_model(model_path)
        
        # 示例：加载一些代码进行比较
        with open(code_results_path, 'r') as f:
            data = json.load(f)
            
        # 选择两个代码样本进行比较
        code1 = data['results'][0]
        code2 = data['results'][1]
        
        # 动态获取问题数量
        num_problems = len(code1['results'])
        
        # 对每个问题分别进行预测
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
            
            # 输出实际性能作为参考
            perf1 = code1['results'][prob_idx]
            perf2 = code2['results'][prob_idx]
            logging.info(f"\nActual performances:")
            logging.info(f"Code 1: {perf1:.4f}")
            logging.info(f"Code 2: {perf2:.4f}")
            logging.info(f"Actually better: Code {1 if perf1 > perf2 else 2}")

if __name__ == "__main__":
    main() 