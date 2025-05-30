# -*- coding: gbk -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import matplotlib.font_manager as fm
import matplotlib as mpl

# 确保使用 Times New Roman 字体
try:
    # 尝试设置 Times New Roman 字体
    mpl.rcParams['font.family'] = 'Times New Roman'
except:
    # 如果找不到 Times New Roman，使用默认衬线字体
    print("Times New Roman font not found, using default serif font")
    mpl.rcParams['font.family'] = 'serif'

def parse_performance_from_file(filename):
    """从验证文件中解析代码表现分数"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
            # 获取倒数第二行的数值
            lines = content.split('\n')
            if len(lines) >= 2:
                return float(lines[-2])
    except:
        return None
    return None

def plot_performance_trends_evaluated(log_dir, fig_path_name):
    """Plot performance trends showing best values achieved so far"""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get all validation files
    files = []
    for root, _, filenames in os.walk(log_dir):
        for f in filenames:
            if f.startswith('problem_iter') and f.endswith('_stdout.txt'):
                files.append(os.path.join(root, f))

    # Parse iteration numbers and response IDs
    all_attempts = []  # Record all attempts including failures
    valid_data = []   # Only contains valid performance data
    
    for f in files:
        match = re.match(r'.*problem_iter(\d+)_response(\d+)\.txt_stdout\.txt', f)
        if match:
            iter_num = int(match.group(1))
            resp_num = int(match.group(2))
            all_attempts.append((iter_num, resp_num))
            
            performance = parse_performance_from_file(f)
            if performance is not None:
                valid_data.append((iter_num, resp_num, performance))
    
    if not valid_data:
        print("No valid data found!")
        return
        
    all_attempts.sort()
    valid_data.sort()
    
    # Track best performance over validation attempts
    performances = []
    best_so_far = float('inf')
    attempt_count = 0
    valid_idx = 0
    
    for attempt in all_attempts:
        attempt_count += 1
        if valid_idx < len(valid_data) and valid_data[valid_idx][:2] == attempt:
            current_perf = valid_data[valid_idx][2]
            print(f"current performance: {current_perf}; best performance: {best_so_far}")
            best_so_far = min(best_so_far, current_perf)
            performances.append(best_so_far)
            valid_idx += 1
        else:
            performances.append(best_so_far if performances else None)
    
    # Filter out None values while keeping correct x-axis count
    valid_performances = [p for p in performances if p is not None]
    valid_indices = [i+1 for i, p in enumerate(performances) if p is not None]
    
    # 计算有效的最大最小值，用于归一化
    min_perf = min(valid_performances)
    max_perf = max(p for p in valid_performances if p != float('inf'))
    
    # 归一化性能值
    def normalize_performance(perf_list):
        return [(p - min_perf) / (max_perf - min_perf) for p in perf_list]
    
    normalized_performances = normalize_performance(valid_performances)
    
    # 设置y轴范围，与plot_comparison_trends保持一致
    y_min = -0.02  # 向下扩展2%
    y_max = 1.02   # 向上扩展2%
    
    # Track top 5 performances and their averages
    top5_averages = []
    for i in range(len(valid_performances)):
        # Get all performances up to current point
        current_perfs = valid_performances[:i+1]
        # Sort and get top 5 (or less if not enough data points)
        top5 = sorted(current_perfs)[:5]
        # Calculate average of available top performances
        avg = sum(top5) / len(top5)
        top5_averages.append(avg)
    
    # Normalize top5 averages
    normalized_top5_averages = normalize_performance(top5_averages)
    
    # Plot original performance line
    ax1.plot(valid_indices, normalized_performances, 'b-', marker='o', label='Best So Far')
    # Add top 5 average line
    # ax1.plot(valid_indices, normalized_top5_averages, 'g--', label='Top 5 Average')
    ax1.legend()
    
    ax1.set_xlabel('Number of Validations (Including Failed Attempts)')
    ax1.set_ylabel('Normalized Performance')
    ax1.set_ylim([y_min, y_max])
    ax1.yaxis.set_major_locator(plt.LinearLocator(6))
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax1.set_title('Best Performance vs Number of Validations')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax1.grid(False, axis='x')
    
    # Track best performance by iteration
    iter_best_performances = {}
    current_best = float('inf')
    
    for iter_num, _, performance in valid_data:
        current_best = min(current_best, performance)
        if iter_num not in iter_best_performances:
            iter_best_performances[iter_num] = current_best
        else:
            iter_best_performances[iter_num] = min(iter_best_performances[iter_num], current_best)
    
    iters = sorted(iter_best_performances.keys())
    best_performances = [iter_best_performances[i] for i in iters]
    
    # 迭代图表也使用归一化的值
    normalized_best_performances = normalize_performance(best_performances)
    
    # For iteration plot, calculate top 5 averages
    iter_top5_averages = {}
    for iter_num in iters:
        # Get all performances up to current iteration
        current_perfs = [iter_best_performances[i] for i in iters if i <= iter_num]
        # Sort and get top 5 (or less if not enough data points)
        top5 = sorted(current_perfs)[:5]
        # Calculate average of available top performances
        avg = sum(top5) / len(top5)
        iter_top5_averages[iter_num] = avg
    
    iter_top5_perf = [iter_top5_averages[i] for i in iters]
    normalized_iter_top5 = normalize_performance(iter_top5_perf)
    
    # Plot original performance line
    ax2.plot(iters, normalized_best_performances, 'r-', marker='o', label='Best So Far')
    # Add top 5 average line
    ax2.plot(iters, normalized_iter_top5, 'g--', label='Top 5 Average')
    ax2.legend()
    
    ax2.set_xlabel('Iteration Number')
    ax2.set_ylabel('Normalized Performance')
    ax2.set_ylim([y_min, y_max])
    ax2.yaxis.set_major_locator(plt.LinearLocator(6))
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax2.set_title('Best Performance vs Iteration Number')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax2.grid(False, axis='x')
    
    # 添加原始值的说明文字
    fig.text(0.02, 0.02, f'Original range: [{min_perf:.3f}, {max_perf:.3f}]', 
             fontsize=8, transform=fig.transFigure)
    
    plt.tight_layout()
    plt.savefig(fig_path_name)
    plt.close()


def plot_performance_trends(log_dir, fig_path_name):
    """Plot performance trends including all generated code (verified and unverified)"""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Get all code files and validation files
    code_files = []
    validation_files = []
    for root, _, filenames in os.walk(log_dir):
        for f in filenames:
            if f.startswith('problem_iter') and f.endswith('.txt'):
                if f.endswith('_stdout.txt'):
                    validation_files.append(os.path.join(root, f))
                else:
                    code_files.append(os.path.join(root, f))

    # Parse iteration numbers and response IDs
    all_codes = []  # Record all code generations
    valid_data = []  # Only contains validated performance data

    for f in code_files:
        match = re.match(r'.*problem_iter(\d+)_response(\d+)\.txt', f)
        if match:
            iter_num = int(match.group(1))
            resp_num = int(match.group(2))
            all_codes.append((iter_num, resp_num))

            # Check if this code was validated
            validation_file = f + '_stdout.txt'
            if os.path.exists(validation_file):
                performance = parse_performance_from_file(validation_file)
                if performance is not None:
                    valid_data.append((iter_num, resp_num, performance))

    if not valid_data:
        print("No valid data found!")
        return

    all_codes.sort()
    valid_data.sort()

    # Track best performance over all code generations
    performances = []
    best_so_far = float('inf')
    valid_idx = 0

    for code in all_codes:
        if valid_idx < len(valid_data) and valid_data[valid_idx][:2] == code:
            # This code was validated
            current_perf = valid_data[valid_idx][2]
            best_so_far = min(best_so_far, current_perf)
            performances.append(best_so_far)
            valid_idx += 1
        else:
            # This code was not validated, keep previous best
            performances.append(best_so_far if performances else None)

    # 计算有效的最大最小值，用于归一化
    valid_perfs = [p for p in performances if p is not None and p != float('inf')]
    min_perf = min(valid_perfs)
    max_perf = max(valid_perfs)
    
    # 归一化函数
    def normalize_performance(perf_list):
        return [(p - min_perf) / (max_perf - min_perf) for p in perf_list if p is not None]
    
    # 归一化性能值
    normalized_performances = normalize_performance(performances)
    
    # 设置y轴范围，与plot_comparison_trends保持一致
    y_min = -0.02
    y_max = 1.02
    
    # Calculate top 5 averages for all codes
    top5_averages = []
    for i in range(len(normalized_performances)):
        current_perfs = performances[:i+1]
        current_perfs = [p for p in current_perfs if p is not None]
        if current_perfs:  # Only calculate if we have valid performances
            top5 = sorted(current_perfs)[:5]
            if top5:  # Additional check to ensure we have values
                avg = sum(top5) / len(top5)
                top5_averages.append(avg)
            else:
                top5_averages.append(None)
        else:
            top5_averages.append(None)
    
    # Filter out None values before normalization
    valid_top5_averages = [x for x in top5_averages if x is not None]
    if valid_top5_averages:  # Only normalize if we have valid averages
        normalized_top5_averages = normalize_performance(valid_top5_averages)
        
        # Plot both lines
        ax1.plot(range(1, len(normalized_performances) + 1), normalized_performances, 'b-', 
                 marker='o', label='Best So Far')
        # ax1.plot(range(1, len(normalized_top5_averages) + 1), normalized_top5_averages, 'g--',
        #          label='Top 5 Average')
        ax1.legend()
    else:
        # If no valid averages, only plot the performances
        ax1.plot(range(1, len(normalized_performances) + 1), normalized_performances, 'b-', 
                 marker='o', label='Best So Far')
        ax1.legend()
    
    ax1.set_xlabel('Number of Generated Codes')
    ax1.set_ylabel('Normalized Performance')
    ax1.set_ylim([y_min, y_max])
    ax1.yaxis.set_major_locator(plt.LinearLocator(6))
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax1.set_title('Best Performance vs Number of Generated Codes')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax1.grid(False, axis='x')
    
    # Track best performance by iteration
    iter_best_performances = {}
    current_best = float('inf')

    # First, record the last code number for each iteration
    iter_code_counts = {}
    for iter_num, resp_num in all_codes:
        if iter_num not in iter_code_counts:
            iter_code_counts[iter_num] = 0
        iter_code_counts[iter_num] += 1

    # Then track best performance for each iteration
    code_count = 0
    for iter_num in sorted(iter_code_counts.keys()):
        codes_in_iter = iter_code_counts[iter_num]
        best_in_iter = performances[code_count:code_count + codes_in_iter]
        best_in_iter = [p for p in best_in_iter if p is not None]
        if best_in_iter:
            iter_best_performances[iter_num] = min(best_in_iter)
        code_count += codes_in_iter

    iters = sorted(iter_best_performances.keys())
    best_performances = [iter_best_performances[i] for i in iters]
    
    # 迭代图表也使用归一化的值
    normalized_best_performances = normalize_performance(best_performances)
    
    # Calculate top 5 averages for iterations
    iter_top5_averages = {}
    for iter_num in iters:
        current_perfs = [iter_best_performances[i] for i in iters if i <= iter_num]
        if current_perfs:  # Only calculate if we have valid performances
            top5 = sorted(current_perfs)[:5]
            if top5:  # Additional check to ensure we have values
                avg = sum(top5) / len(top5)
                iter_top5_averages[iter_num] = avg
    
    if iter_top5_averages:  # Only proceed if we have valid averages
        iter_top5_perf = [iter_top5_averages[i] for i in iters if i in iter_top5_averages]
        if iter_top5_perf:  # Additional check before normalization
            normalized_iter_top5 = normalize_performance(iter_top5_perf)
            
            # Plot both lines
            ax2.plot(iters, normalized_best_performances, 'r-', marker='o', label='Best So Far')
            ax2.plot(iters, normalized_iter_top5, 'g--', label='Top 5 Average')
            ax2.legend()
        else:
            # If no valid averages, only plot the performances
            ax2.plot(iters, normalized_best_performances, 'r-', marker='o', label='Best So Far')
            ax2.legend()
    else:
        # If no valid averages, only plot the performances
        ax2.plot(iters, normalized_best_performances, 'r-', marker='o', label='Best So Far')
        ax2.legend()
    
    ax2.set_xlabel('Iteration Number')
    ax2.set_ylabel('Normalized Performance')
    ax2.set_ylim([y_min, y_max])
    ax2.yaxis.set_major_locator(plt.LinearLocator(6))
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax2.set_title('Best Performance vs Iteration Number')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax2.grid(False, axis='x')
    
    # 添加原始值的说明文字
    fig.text(0.02, 0.02, f'Original range: [{min_perf:.3f}, {max_perf:.3f}]', 
             fontsize=8, transform=fig.transFigure)
    
    plt.tight_layout()
    plt.savefig(fig_path_name)
    plt.close()


def plot_comparison_trends(log_dir1, log_dir2, label1="Experiment 1", label2="Experiment 2", 
                         output_filename="comparison_trends.png"):
    """Plot comparison of performance trends between two experiments"""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Process data for both experiments
    def get_experiment_data(log_dir):
        """获取实验数据，包括所有代码文件和验证文件"""
        code_files = []
        validation_files = []
        for root, _, filenames in os.walk(log_dir):
            for f in filenames:
                if f.startswith('problem_iter') and f.endswith('.txt'):
                    if f.endswith('_stdout.txt'):
                        validation_files.append(os.path.join(root, f))
                    else:
                        code_files.append(os.path.join(root, f))
        
        all_attempts = []  # 所有代码生成尝试
        valid_data = []   # 验证数据
        
        for f in code_files:
            match = re.match(r'.*problem_iter(\d+)_response(\d+)\.txt', f)
            if match:
                iter_num = int(match.group(1))
                resp_num = int(match.group(2))
                all_attempts.append((iter_num, resp_num))
                
                # 检查是否有对应的验证文件
                validation_file = f + '_stdout.txt'
                if os.path.exists(validation_file):
                    performance = parse_performance_from_file(validation_file)
                    if performance is not None:
                        valid_data.append((iter_num, resp_num, performance))
        
        return sorted(all_attempts), sorted(valid_data)
    
    # Get data for both experiments
    all_attempts1, valid_data1 = get_experiment_data(log_dir1)
    all_attempts2, valid_data2 = get_experiment_data(log_dir2)
    
    if not valid_data1 or not valid_data2:
        print("No valid data found in one or both experiments!")
        return
    
    # Plot validation trends
    def get_performance_trend(all_attempts, valid_data, is_predictor=False):
        """获取性能趋势
        is_predictor: 是否是带预测器的方法(_p结尾的方法)
        返回：
        - indices: 所有代码点的位置
        - values: 对应的性能值
        - valid_points: 实际验证过的点的位置
        """
        performances = []
        valid_points = []  # 记录实际验证过的点的位置
        best_so_far = float('inf')
        valid_idx = 0
        
        for i, attempt in enumerate(all_attempts, 1):
            # 检查当前代码是否有验证结果
            has_validation = False
            current_perf = None
            
            while valid_idx < len(valid_data) and valid_data[valid_idx][:2] <= attempt:
                if valid_data[valid_idx][:2] == attempt:
                    has_validation = True
                    current_perf = valid_data[valid_idx][2]
                valid_idx += 1
            
            if has_validation:
                # 当前代码被验证过
                best_so_far = min(best_so_far, current_perf)
                performances.append((i, best_so_far))
                valid_points.append(i)  # 记录验证点的位置
            elif not is_predictor:
                # 非预测器方法：记录所有点的值，但不标记为验证点
                performances.append((i, best_so_far))
        
        # 分离x和y值
        if performances:
            indices, values = zip(*performances)
            return list(indices), list(values), valid_points
        return [], [], []
    
    # Plot validation comparison
    indices1, performances1, valid_points1 = get_performance_trend(all_attempts1, valid_data1, 
                                                                 is_predictor='_p' in label1)
    indices2, performances2, valid_points2 = get_performance_trend(all_attempts2, valid_data2, 
                                                                 is_predictor='_p' in label2)
    
    # 计算两个实验的整体最大最小值
    all_performances = performances1 + performances2
    min_perf = min(all_performances)
    max_perf = max(p for p in all_performances if p != float('inf'))
    
    # 将性能值归一化到0-1区间
    def normalize_performance(perf_list):
        return [(p - min_perf) / (max_perf - min_perf) for p in perf_list]
    
    # 归一化两个实验的性能值
    normalized_performances1 = normalize_performance(performances1)
    normalized_performances2 = normalize_performance(performances2)
    
    # 设置y轴范围，稍微扩展一点以避免数据点贴边
    y_min = -0.02  # 向下扩展2%
    y_max = 1.02   # 向上扩展2%
    
    # Calculate and plot top 5 averages for both experiments
    def get_top5_averages(performances):
        top5_avgs = []
        for i in range(len(performances)):
            current_perfs = performances[:i+1]
            top5 = sorted(current_perfs)[:5]
            avg = sum(top5) / len(top5)
            top5_avgs.append(avg)
        return normalize_performance(top5_avgs)
    
    # Calculate top 5 averages
    normalized_top5_1 = get_top5_averages(performances1)
    normalized_top5_2 = get_top5_averages(performances2)
    
    # Plot all lines
    ax1.plot(indices1, normalized_performances1, 'b-', label=f'{label1} Best')
    # ax1.plot(indices1, normalized_top5_1, 'b--', label=f'{label1} Top 5 Avg')
    ax1.plot(indices2, normalized_performances2, 'r-', label=f'{label2} Best')
    # ax1.plot(indices2, normalized_top5_2, 'r--', label=f'{label2} Top 5 Avg')
    
    # 再标记验证点
    if valid_points1:
        ax1.plot([i for i in valid_points1], 
                [normalized_performances1[indices1.index(i)] for i in valid_points1], 
                'bo', markersize=6)
    if valid_points2:
        ax1.plot([i for i in valid_points2], 
                [normalized_performances2[indices2.index(i)] for i in valid_points2], 
                'rs', markersize=6)
    
    ax1.set_xlabel('Number of Code Attempts')
    ax1.set_ylabel('Normalized Performance')
    ax1.set_ylim([y_min, y_max])
    ax1.yaxis.set_major_locator(plt.LinearLocator(6))  # 减少刻度数量到6个
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax1.set_title('Best Performance vs Number of Code Attempts')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax1.grid(False, axis='x')
    ax1.legend()
    
    # Plot iteration trends
    def get_iteration_trend(valid_data):
        iter_best_performances = {}
        current_best = float('inf')
        
        for iter_num, _, performance in valid_data:
            current_best = min(current_best, performance)
            if iter_num not in iter_best_performances:
                iter_best_performances[iter_num] = current_best
            else:
                iter_best_performances[iter_num] = min(iter_best_performances[iter_num], current_best)
        
        iters = sorted(iter_best_performances.keys())
        best_performances = [iter_best_performances[i] for i in iters]
        # 归一化迭代性能值
        normalized_best = normalize_performance(best_performances)
        return iters, normalized_best
    
    # Plot iteration comparison
    iters1, best_perfs1 = get_iteration_trend(valid_data1)
    iters2, best_perfs2 = get_iteration_trend(valid_data2)
    
    # Calculate and plot iteration-based top 5 averages
    def get_iteration_top5_trend(valid_data):
        iter_best = {}
        for iter_num, _, performance in valid_data:
            if iter_num not in iter_best:
                iter_best[iter_num] = []
            iter_best[iter_num].append(performance)
        
        iters = sorted(iter_best.keys())
        top5_avgs = []
        for i, iter_num in enumerate(iters):
            current_perfs = []
            for j in range(i+1):
                current_perfs.extend(iter_best[iters[j]])
            top5 = sorted(current_perfs)[:5]
            avg = sum(top5) / len(top5)
            top5_avgs.append(avg)
        
        return iters, normalize_performance(top5_avgs)
    
    # Plot iteration trends with top 5 averages
    iters1, top5_perfs1 = get_iteration_top5_trend(valid_data1)
    iters2, top5_perfs2 = get_iteration_top5_trend(valid_data2)
    
    ax2.plot(iters1, best_perfs1, 'b-', marker='o', label=f'{label1} Best')
    ax2.plot(iters1, top5_perfs1, 'b--', label=f'{label1} Top 5 Avg')
    ax2.plot(iters2, best_perfs2, 'r-', marker='s', label=f'{label2} Best')
    ax2.plot(iters2, top5_perfs2, 'r--', label=f'{label2} Top 5 Avg')
    
    ax2.set_xlabel('Iteration Number')
    ax2.set_ylabel('Normalized Performance')
    ax2.set_ylim([y_min, y_max])
    ax2.yaxis.set_major_locator(plt.LinearLocator(6))
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax2.set_title('Best Performance vs Iteration Number')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax2.grid(False, axis='x')
    ax2.legend()
    
    # 添加原始值的说明文字
    fig.text(0.02, 0.02, f'Original range: [{min_perf:.3f}, {max_perf:.3f}]', 
             fontsize=8, transform=fig.transFigure)
    
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


def plot_time_performance_trends(log_dir, fig_path_name):
    """Plot performance trends over time based on obj_time.log file"""
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Read obj_time.log file
    log_file = os.path.join(log_dir, 'obj_time.log')
    if not os.path.exists(log_file):
        print(f"No obj_time.log found in {log_dir}")
        return
        
    times = []
    best_perfs = []
    top5_avgs = []
    
    with open(log_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                time = float(parts[0])
                best_perf = float(parts[1]) if parts[1] != 'inf' else float('inf')
                top5_avg = float(parts[2]) if parts[2] != 'inf' else float('inf')
                
                times.append(time)
                best_perfs.append(best_perf)
                top5_avgs.append(top5_avg)
    
    if not times:
        print("No valid data found in obj_time.log")
        return
        
    # Filter out inf values
    valid_perfs = [(t, p) for t, p in zip(times, best_perfs) if p != float('inf')]
    valid_avgs = [(t, a) for t, a in zip(times, top5_avgs) if a != float('inf')]
    
    if valid_perfs:
        valid_times, valid_best_perfs = zip(*valid_perfs)
        plt.plot(valid_times, valid_best_perfs, 'b-', marker='o', label='Best So Far')
        
    if valid_avgs:
        avg_times, valid_top5_avgs = zip(*valid_avgs)
        plt.plot(avg_times, valid_top5_avgs, 'g--', label='Top 5 Average')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Performance')
    # plt.title('Performance vs Time')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.grid(False, axis='x')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(fig_path_name)
    plt.close()


def plot_time_comparison_trends(log_dir1, log_dir2, label1="Experiment 1", label2="Experiment 2",
                                output_filename="time_comparison_trends.png"):
    """Plot comparison of performance trends between two experiments over time"""
    # 设置全局字体大小和粗细
    plt.rcParams.update({
        'font.family': 'Times New Roman',  # 设置Times New Roman字体
        'font.size': 38,  # 增大基础字体大小
        'font.weight': 'bold',
        'axes.titlesize': 42,  # 增大标题字体大小
        'axes.labelsize': 40,  # 增大轴标签字体大小
        'xtick.labelsize': 38,  # 增大x轴刻度标签大小
        'ytick.labelsize': 38,  # 增大y轴刻度标签大小
        'legend.fontsize': 38,  # 增大图例字体大小
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
    })

    # Create figure with larger size
    plt.figure(figsize=(16, 12))  # 增大图形尺寸
    
    def read_time_data(log_dir):
        """Read time and performance data from obj_time.log"""
        log_file = os.path.join(log_dir, 'obj_time.log')
        if not os.path.exists(log_file):
            print(f"No obj_time.log found in {log_dir}")
            return [], [], []
            
        times = []
        best_perfs = []
        top5_avgs = []
        
        with open(log_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    time = float(parts[0])
                    best_perf = float(parts[1]) if parts[1] != 'inf' else float('inf')
                    top5_avg = float(parts[2]) if parts[2] != 'inf' else float('inf')
                    
                    times.append(time)
                    best_perfs.append(best_perf)
                    top5_avgs.append(top5_avg)
                    
        return times, best_perfs, top5_avgs
    
    # Read data from both experiments
    times1, best_perfs1, top5_avgs1 = read_time_data(log_dir1)
    times2, best_perfs2, top5_avgs2 = read_time_data(log_dir2)
    
    if not times1 or not times2:
        print("No valid data found in one or both experiments!")
        return
        
    # Filter out inf values and combine data for normalization
    valid_perfs1 = [p for p in best_perfs1 if p != float('inf')]
    valid_perfs2 = [p for p in best_perfs2 if p != float('inf')]
    all_valid_perfs = valid_perfs1 + valid_perfs2
    
    if not all_valid_perfs:
        print("No valid performance values found!")
        return
        
    # Calculate normalization range
    min_perf = min(all_valid_perfs)
    max_perf = max(all_valid_perfs)
    
    # Normalize function
    def normalize_performance(perf_list):
        return [(p - min_perf) / (max_perf - min_perf) if p != float('inf') else None 
                for p in perf_list]
    
    # Normalize performances
    norm_best_perfs1 = normalize_performance(best_perfs1)
    norm_best_perfs2 = normalize_performance(best_perfs2)
    norm_top5_avgs1 = normalize_performance(top5_avgs1)
    norm_top5_avgs2 = normalize_performance(top5_avgs2)
    
    # Filter out None values while keeping corresponding times
    valid_data1 = [(t, p) for t, p in zip(times1, norm_best_perfs1) if p is not None]
    valid_data2 = [(t, p) for t, p in zip(times2, norm_best_perfs2) if p is not None]
    valid_avg1 = [(t, p) for t, p in zip(times1, norm_top5_avgs1) if p is not None]
    valid_avg2 = [(t, p) for t, p in zip(times2, norm_top5_avgs2) if p is not None]
    
    # Plot the data
    if valid_data1:
        times1, perfs1 = zip(*valid_data1)
        plt.plot(times1, perfs1, 'b-', marker='o', label=f'{label1} Best')
    if valid_data2:
        times2, perfs2 = zip(*valid_data2)
        plt.plot(times2, perfs2, 'r-', marker='s', label=f'{label2} Best')
    if valid_avg1:
        avg_times1, avg_perfs1 = zip(*valid_avg1)
        plt.plot(avg_times1, avg_perfs1, 'b--', label=f'{label1} Top 5 Avg')
    if valid_avg2:
        avg_times2, avg_perfs2 = zip(*valid_avg2)
        plt.plot(avg_times2, avg_perfs2, 'r--', label=f'{label2} Top 5 Avg')
    
    # Set y-axis range
    y_min = -0.02  # Extend down by 2%
    y_max = 1.02   # Extend up by 2%
    plt.ylim([y_min, y_max])
    
    # 设置轴标签和标题，使用更大字号
    plt.xlabel('Time (seconds)', fontsize=40, labelpad=20, weight='bold')
    plt.ylabel('Normalized Performance', fontsize=40, labelpad=20, weight='bold')
    
    # 加粗坐标轴和刻度线
    plt.gca().spines['bottom'].set_linewidth(3.0)
    plt.gca().spines['left'].set_linewidth(3.0)
    plt.gca().spines['top'].set_linewidth(3.0)
    plt.gca().spines['right'].set_linewidth(3.0)
    plt.gca().tick_params(width=3.0, length=12)
    
    # 设置图例，调整位置和大小
    plt.legend(loc='upper right', fontsize=38, frameon=True,
              facecolor='white', edgecolor='black',
              bbox_to_anchor=(1.0, 0.98),  # 将图例位置从0.95上移至0.98
              borderpad=0.4,  # 减小边框padding
              handlelength=2,  # 减小图例标记长度
              handletextpad=0.4,  # 减小文本间距
              borderaxespad=0.2,  # 减小轴边距
              labelspacing=0.4)  # 减小标签间距

    # Add original range text with larger font
    plt.figtext(0.02, 0.02, f'Original range: [{min_perf:.3f}, {max_perf:.3f}]',
                fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()


def plot_time_multi_comparison_trends(log_dirs_dict, output_filename="time_multi_comparison_trends.png",
                                      x_range=None):
    """Plot comparison of performance trends among multiple experiments over time"""
    # 设置全局字体大小和粗细
    plt.rcParams.update({
        'font.family': 'Times New Roman',  # 设置Times New Roman字体
        'font.size': 42,  # 增大基础字体大小
        'font.weight': 'bold',
        'axes.titlesize': 50,  # 增大标题字体大小
        'axes.labelsize': 50,  # 增大轴标签字体大小
        'xtick.labelsize': 45,  # 增大x轴刻度标签大小
        'ytick.labelsize': 45,  # 增大y轴刻度标签大小
        'legend.fontsize': 38,  # 增大图例字体大小
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
    })

    # 创建更大尺寸的图表
    plt.figure(figsize=(16, 12))

    # Define styles for different methods with larger markers and lines
    styles = {
        'reevo': {
            'color': '#3498db',  # 使用更专业的配色
            'marker': 'o',
            'markersize': 12,  # 增大标记大小
            'linewidth': 5  # 增大线宽
        },
        'nerm': {
            'color': '#e74c3c',
            'marker': 's',
            'markersize': 12,
            'linewidth': 5
        },
        'eoh': {
            'color': '#2ecc71',
            'marker': '^',
            'markersize': 12,
            'linewidth': 5
        },
        'default': {
            'color': 'black',
            'marker': 'x',
            'markersize': 12,
            'linewidth': 5
        }
    }

    # Read data from all experiments
    all_data = {}
    for label, log_dir in log_dirs_dict.items():
        log_file = os.path.join(log_dir, 'obj_time.log')
        if not os.path.exists(log_file):
            print(f"No obj_time.log found in {log_dir}")
            continue
        
        times = []
        best_perfs = []
        top5_avgs = []
        
        with open(log_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    time = float(parts[0])
                    best_perf = float(parts[1]) if parts[1] != 'inf' else float('inf')
                    top5_avg = float(parts[2]) if parts[2] != 'inf' else float('inf')
                    
                    times.append(time)
                    best_perfs.append(best_perf)
                    top5_avgs.append(top5_avg)
        
        if times:
            all_data[label] = (times, best_perfs, top5_avgs)
    
    if not all_data:
        print("No valid data found in any experiment!")
        return
    
    # Calculate normalization range
    all_valid_perfs = [p for _, best_perfs, _ in all_data.values() for p in best_perfs if p != float('inf')]
    min_perf = min(all_valid_perfs)
    max_perf = max(all_valid_perfs)
    #
    # max_perf = 16.78930
    # Normalize function
    def normalize_performance(perf_list):
        return [(p - min_perf) / (max_perf - min_perf) if p != float('inf') else None 
                for p in perf_list]
    
    # Plot data for each experiment
    for label, (times, best_perfs, top5_avgs) in all_data.items():
        # Normalize performances
        norm_best_perfs = normalize_performance(best_perfs)
        norm_top5_avgs = normalize_performance(top5_avgs)
        
        # Filter out None values while keeping corresponding times
        valid_data = [(t, p) for t, p in zip(times, norm_best_perfs) if p is not None]
        valid_avg = [(t, p) for t, p in zip(times, norm_top5_avgs) if p is not None]
        
        # Filter data points based on x_range if specified
        if x_range is not None:
            min_x, max_x = x_range
            valid_data = [(t, p) for t, p in valid_data if min_x <= t <= max_x]
            valid_avg = [(t, p) for t, p in valid_avg if min_x <= t <= max_x]
        
        # Get style for the current method
        method_style = styles.get(label.lower(), styles['default'])
        
        # Plot best performance
        if valid_data:
            plot_times, plot_perfs = zip(*valid_data)
            plt.plot(plot_times, plot_perfs, 
                    color=method_style['color'],
                    linestyle='-',
                    marker=method_style['marker'],
                    markerfacecolor='white',  # 设置标记内部为白色
                    markersize=method_style['markersize'],
                    linewidth=method_style['linewidth'],
                    markeredgewidth=3,  # 增加标记边缘宽度
                    label=f'{label} Best')
        
        # Plot top 5 average
        if valid_avg:
            avg_times, avg_perfs = zip(*valid_avg)
            plt.plot(avg_times, avg_perfs,
                    color=method_style['color'],
                    linestyle='--',
                    marker=method_style['marker'],
                    markerfacecolor='white',
                    markersize=method_style['markersize'],
                    linewidth=method_style['linewidth'],
                    markeredgewidth=3,
                    label=f'{label} Top 5 Avg')
    
    # Set axis ranges
    y_min = -0.02
    y_max = 1.02
    plt.ylim([y_min, y_max])
    
    if x_range is not None:
        plt.xlim(x_range)
    
    # 设置轴标签和标题，使用更大字号
    plt.xlabel('Time (seconds)', fontsize=50, labelpad=20, weight='bold')
    plt.ylabel('Normalized Performance', fontsize=50, labelpad=20, weight='bold')
    
    # 加粗坐标轴和刻度线
    plt.gca().spines['bottom'].set_linewidth(3.0)
    plt.gca().spines['left'].set_linewidth(3.0)
    plt.gca().spines['top'].set_linewidth(3.0)
    plt.gca().spines['right'].set_linewidth(3.0)
    
    # 加粗刻度线
    plt.gca().tick_params(width=3.0, length=12)
    
    # 设置网格线样式
    plt.grid(True, axis='y', linestyle='--', alpha=0.7, linewidth=2.5)
    plt.grid(False, axis='x')
    
    # 设置图例，调整位置和大小
    plt.legend(loc='upper right', fontsize=38, frameon=True,
              facecolor='white', edgecolor='black',
              bbox_to_anchor=(1.0, 1.0),  # 将图例位置从0.95上移至0.98
              borderpad=0.4,  # 减小边框padding
              handlelength=2,  # 减小图例标记长度
              handletextpad=0.4,  # 减小文本间距
              borderaxespad=0.2,  # 减小轴边距
              labelspacing=0.4)  # 减小标签间距

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)  # 增加DPI以提高图片质量
    plt.close()


if __name__ == '__main__':
    pictures_dir = './pictures/'
    if not os.path.exists(pictures_dir):
        os.makedirs(pictures_dir)

    # coevolve reflect predictor (predictor reevo)
    log_c_r_p_reevo_p_run1 = 'backup/finalized/gpt3.5/coevolve_reflect_predictor/coevolve_pop11_popsize12_reevo_max_fe220/run1/2025-01-09_13-54-31/'
    log_c_r_p_reevo_p_run2 = 'backup/finalized/gpt3.5/coevolve_reflect_predictor/coevolve_pop11_popsize12_reevo_max_fe220/run2/2025-01-09_14-25-16/'
    log_c_r_p_reevo_p_run3 = 'backup/finalized/gpt3.5/coevolve_reflect_predictor/coevolve_pop11_popsize12_reevo_max_fe220/run3/2025-01-09_15-06-06/'

    # coevolve reflect predictor (full reevo)
    log_c_r_p_reevo_full_run1 = 'backup/finalized/gpt3.5/coevolve_reflect_predictor_reevo_full/run1/2025-01-05_20-30-59/'
    log_c_r_p_reevo_full_run2 = 'backup/finalized/gpt3.5/coevolve_reflect_predictor_reevo_full/run2/2025-01-05_17-05-56/'
    log_c_r_p_reevo_full_run3 = 'backup/finalized/gpt3.5/coevolve_reflect_predictor_reevo_full/run3/2025-01-05_19-10-51/'

    # reevo predictor
    log_reevo_predictor_run1 = 'backup/finalized/gpt3.5/reevo/predictor_max_fe220_236/run1/2025-01-08_13-02-33/'
    log_reevo_predictor_run2 = 'backup/finalized/gpt3.5/reevo/predictor_max_fe220_236/run2/2025-01-08_11-47-27/'
    log_reevo_predictor_run3 = 'backup/finalized/gpt3.5/reevo/predictor_max_fe220_236/run3/2025-01-08_12-21-15/'

    # reevo
    log_reevo_run1 = 'backup/finalized/gpt3.5/reevo/max_fe220_236/run1/2025-01-07_17-04-11/'
    log_reevo_run2 = 'backup/finalized/gpt3.5/reevo/max_fe220_236/run2/2025-01-07_18-02-24/'
    log_reevo_run3 = 'backup/finalized/gpt3.5/reevo/max_fe220_236/run3/2025-01-08_14-15-57/'

    # # 将所有日志路径和对应的标识符放入字典中
    # log_paths = {
    #     'c_r_p_reevo_p_run1': log_c_r_p_reevo_p_run1,
    #     'c_r_p_reevo_p_run2': log_c_r_p_reevo_p_run2,
    #     'c_r_p_reevo_p_run3': log_c_r_p_reevo_p_run3,
    #     'c_r_p_reevo_full_run1': log_c_r_p_reevo_full_run1,
    #     'c_r_p_reevo_full_run2': log_c_r_p_reevo_full_run2,
    #     'c_r_p_reevo_full_run3': log_c_r_p_reevo_full_run3,
    #     'reevo_predictor_run1': log_reevo_predictor_run1,
    #     'reevo_predictor_run2': log_reevo_predictor_run2,
    #     'reevo_predictor_run3': log_reevo_predictor_run3,
    #     'reevo_run1': log_reevo_run1,
    #     'reevo_run2': log_reevo_run2,
    #     'reevo_run3': log_reevo_run3
    # }
    #
    # # 批量生成图片
    # for identifier, log_path in log_paths.items():
    #     # 生成仅包含已评估代码的趋势图
    #     evaluated_fig_name = f'trend_evaluated_{identifier}.png'
    #     plot_performance_trends_evaluated(log_path, pictures_dir + evaluated_fig_name)
    #     print(f"Generated {evaluated_fig_name}")
    #
    #     # 生成包含所有代码的趋势图
    #     all_codes_fig_name = f'trend_{identifier}.png'
    #     plot_performance_trends(log_path, pictures_dir + all_codes_fig_name)
    #     print(f"Generated {all_codes_fig_name}")
    #
    # # 添加实验对比
    # comparison_pairs = [
    #     # 对比run1的实验
    #     (log_c_r_p_reevo_p_run1, log_reevo_run1,
    #      "C-R-P-ReEvo-P Run1", "ReEvo Run1",
    #      "comparison_crp_vs_reevo_run1.png"),
    #     # 对比run2的实验
    #     (log_c_r_p_reevo_p_run2, log_reevo_run2,
    #      "C-R-P-ReEvo-P Run2", "ReEvo Run2",
    #      "comparison_crp_vs_reevo_run2.png"),
    #     # 对比run3的实验
    #     (log_c_r_p_reevo_p_run3, log_reevo_run3,
    #      "C-R-P-ReEvo-P Run3", "ReEvo Run3",
    #      "comparison_crp_vs_reevo_run3.png")
    # ]
    #
    # # 生成对比图
    # for log_dir1, log_dir2, label1, label2, output_name in comparison_pairs:
    #     plot_comparison_trends(
    #         log_dir1,
    #         log_dir2,
    #         label1=label1,
    #         label2=label2,
    #         output_filename=pictures_dir + output_name
    #     )
    #     print(f"Generated comparison plot: {output_name}")

    pictures_dir = './pictures/threshold/'
    if not os.path.exists(pictures_dir):
        os.makedirs(pictures_dir)

    # # TSP200
    # coevolve reflect predictor (predictor reevo)
    log_c_r_p_reevo_p_th_run1 = 'backup/threshold/tsp200/coevolve/run4/2025-01-24_00-08-08/'
    log_reevo_th_run1 = 'backup/threshold/tsp200/reevo/run3/2025-01-23_13-52-33/'
    log_eoh_th_run1 = 'backup/threshold/tsp200/eoh/run1/2025-01-24_18-48-15/'

    # # ## TSP300
    # log_c_r_p_reevo_p_th_run1 = 'backup/threshold/tsp300/coevolve/run1/2025-01-25_03-41-29/'
    # log_reevo_th_run1 = 'backup/threshold/tsp300/reevo/run1/2025-01-25_04-20-12/'
    # # log_reevo_th_run1 = 'backup/threshold/tsp300/reevo/run2/2025-01-28_14-06-01/'
    # log_eoh_th_run1 = 'backup/threshold/tsp300/eoh/2025-01-25_13-34-21/'

    # # ## TSP400
    # log_c_r_p_reevo_p_th_run1 = 'backup/threshold/tsp400/coevolve/run3/2025-01-26_09-08-52/'
    # log_reevo_th_run1 = 'backup/threshold/tsp400/reevo/run3/2025-01-25_20-53-58/'
    # # log_reevo_th_run1 = 'backup/threshold/tsp400/reevo/run6-14 9237/2025-01-28_10-41-39/' # 14.9237
    # log_eoh_th_run1 = 'backup/threshold/tsp400/eoh/2025-01-25_13-31-19/'

    # TSP500
    # log_c_r_p_reevo_p_th_run1 = 'backup/threshold/tsp500/coevolve_r_p/run2/2025-01-25_16-52-15/'
    # log_reevo_th_run1 = 'backup/threshold/tsp500/reevo/2025-01-23_14-11-00/'
    # log_eoh_th_run1 = 'backup/threshold/tsp500/eoh/2025-01-25_03-55-24/'

    # 添加实验对比
    comparison_pairs = [
        # 对比run1的实验
        (log_c_r_p_reevo_p_th_run1, log_reevo_th_run1,
         "NeRM", "ReEvo",
         "threshold_comparison_NeRM_vs_ReEvo.png"),
    ]

    # # 生成对比图
    # for log_dir1, log_dir2, label1, label2, output_name in comparison_pairs:
    #     plot_comparison_trends(
    #         log_dir1,
    #         log_dir2,
    #         label1=label1,
    #         label2=label2,
    #         output_filename=pictures_dir + output_name
    #     )
    #     print(f"Generated comparison plot: {output_name}")

    # Add time-based visualization
    pictures_dir = './pictures/threshold/'
    if not os.path.exists(pictures_dir):
        os.makedirs(pictures_dir)
        
    # Generate time-based performance plots
    log_paths = {
        'NeRM': log_c_r_p_reevo_p_th_run1,
        'ReEvo': log_reevo_th_run1
    }
    
    for identifier, log_path in log_paths.items():
        time_fig_name = f'time_trend_{identifier}.png'
        plot_time_performance_trends(log_path, pictures_dir + time_fig_name)
        print(f"Generated {time_fig_name}")

    # Add time-based comparison visualization
    pictures_dir = './pictures/threshold/'
    if not os.path.exists(pictures_dir):
        os.makedirs(pictures_dir)
        
    # Generate time-based comparison plots
    time_comparison_pairs = [
        (log_c_r_p_reevo_p_th_run1, log_reevo_th_run1,
         "NeRM", "ReEvo",
         "threshold_time_comparison_NeRM_vs_ReEvo.png")
    ]
    
    for log_dir1, log_dir2, label1, label2, output_name in time_comparison_pairs:
        plot_time_comparison_trends(
            log_dir1,
            log_dir2,
            label1=label1,
            label2=label2,
            output_filename=pictures_dir + output_name
        )
        print(f"Generated time comparison plot: {output_name}")

    # Add multi-comparison visualization
    pictures_dir = './pictures/threshold/'
    if not os.path.exists(pictures_dir):
        os.makedirs(pictures_dir)



    # ------------------------------ multi plot
    # Example usage of multi-comparison plot
    log_dirs_dict = {
        'NeRM': log_c_r_p_reevo_p_th_run1,
        'ReEvo': log_reevo_th_run1,
        'EoH': log_eoh_th_run1  # Replace with actual EOH log directory
    }

    # [6000, 6000, 9000, 12710]
    plot_time_multi_comparison_trends(
        log_dirs_dict,
        output_filename=pictures_dir + 'threshold_time_comparison_multi.png',
        x_range=(0, 6000)  # Specify the x-axis range
    )
    print("Generated multi-comparison time plot")


