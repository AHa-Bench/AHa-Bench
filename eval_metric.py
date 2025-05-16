import json
from collections import defaultdict
import jiwer
import os

def normalize(s):
    s = s.strip().lower()
    if 'yes' in s or '是' in s:
        return 'yes'
    if 'no' in s or '否' in s:
        return 'no'
    return s

def get_task_type(type_str):
    return type_str.split('_')[0]

def compute_wer(ref, hyp):
    return jiwer.wer(ref.strip().lower(), hyp.strip().lower())

def get_sample_key(index):
    return '_'.join(index.split('_')[:2])

# 配置base_path和file_prefix列表
base_prefix_list = [
    # (base_path, file_prefix)
]

num_files = 8  # 每组有8个文件

# 结果表格数据结构
all_results = []

for base_path, file_prefix in base_prefix_list:
    file_list = [os.path.join(base_path, f"{file_prefix}{i}_{i}.jsonl") for i in range(num_files)]
    all_task_acc = defaultdict(list)
    all_avg_acc = []
    all_bias = defaultdict(list)
    all_consistency_true = defaultdict(list)
    all_consistency_wrong = defaultdict(list)

    for filename in file_list:
        results = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                dataa = json.loads(line)
                if dataa['index']=='40_10_3' or dataa['index']=='40_10_4':
                    print(dataa['question'])
                    continue
                results.append(dataa)

        task_stats = defaultdict(lambda: {"correct": 0, "total": 0, "items": []})
        task_sample_group = defaultdict(lambda: defaultdict(list))

        for item in results:
            type_str = item['type']
            task_type = get_task_type(type_str)
            answer = normalize(item['answer'])
            prediction = item['prediction']
            prediction_match = normalize(item['prediction_match'])
            is_asr = 'asr' in type_str.lower()
            correct = False

            if is_asr:
                wer = compute_wer(item['answer'], item['prediction'])
                correct = wer < 0.1
            else:
                correct = prediction_match == answer

            task_stats[task_type]["correct"] += int(correct)
            task_stats[task_type]["total"] += 1
            task_stats[task_type]["items"].append({
                "answer": answer,
                "prediction": prediction,
                "prediction_match": prediction_match,
                "correct": correct,
                "index": item['index']
            })

            sample_key = get_sample_key(item['index'])
            task_sample_group[task_type][sample_key].append({
                "answer": answer,
                "prediction_match": prediction_match,
                "correct": correct
            })

        total_correct, total_total = 0, 0
        for task_type, stat in task_stats.items():
            acc = stat["correct"] / stat["total"] if stat["total"] > 0 else 0
            all_task_acc[task_type].append(acc)
            if ('random' in base_path) and ('asr' in task_type.lower()):
                continue
            total_correct += stat["correct"]
            total_total += stat["total"]
        avg_acc = total_correct / total_total if total_total > 0 else 0
        all_avg_acc.append(avg_acc)

        for task_type, stat in task_stats.items():
            if 'asr' in task_type.lower():
                continue
            FP = 0
            FN = 0
            P = 0
            N = 0
            for item in stat["items"]:
                answer = item["answer"]
                prediction_match = item["prediction_match"]
                if answer == "yes":
                    P += 1
                    if prediction_match == "no":
                        FN += 1
                elif answer == "no":
                    N += 1
                    if prediction_match == "yes":
                        FP += 1
            FNR = FN / P if P > 0 else 0
            FPR = FP / N if N > 0 else 0
            bias = FNR - FPR
            all_bias[task_type].append(bias)

        for task_type, sample_group in task_sample_group.items():
            total_samples = 0
            consistent_true = 0
            consistent_wrong = 0
            for sample_key, items in sample_group.items():
                total_samples += 1
                correct_list = [item['correct'] for item in items]
                if all(correct_list):
                    consistent_true += 1
                elif not any(correct_list):
                    consistent_wrong += 1
            cons_true = consistent_true / total_samples if total_samples > 0 else 0
            cons_wrong = consistent_wrong / total_samples if total_samples > 0 else 0
            all_consistency_true[task_type].append(cons_true)
            all_consistency_wrong[task_type].append(cons_wrong)

    # 汇总平均，保存到all_results
    task_types = list(all_task_acc.keys() | all_bias.keys() | all_consistency_true.keys())
    task_types = sorted(task_types)
    acc_row = {}
    bias_row = {}
    cons_true_row = {}
    cons_wrong_row = {}
    for task_type in task_types:
        acc = sum(all_task_acc[task_type]) / len(all_task_acc[task_type]) if task_type in all_task_acc else None
        bias = sum(all_bias[task_type]) / len(all_bias[task_type]) if task_type in all_bias else None
        cons_true = sum(all_consistency_true[task_type]) / len(all_consistency_true[task_type]) if task_type in all_consistency_true else None
        cons_wrong = sum(all_consistency_wrong[task_type]) / len(all_consistency_wrong[task_type]) if task_type in all_consistency_wrong else None
        acc_row[task_type] = acc
        bias_row[task_type] = bias
        cons_true_row[task_type] = cons_true
        cons_wrong_row[task_type] = cons_wrong
    acc_row['平均'] = sum(all_avg_acc) / len(all_avg_acc)
    bias_row['平均'] = sum([v for v in bias_row.values() if v is not None]) / len([v for v in bias_row.values() if v is not None])
    cons_true_row['平均'] = sum([v for v in cons_true_row.values() if v is not None]) / len([v for v in cons_true_row.values() if v is not None])
    cons_wrong_row['平均'] = sum([v for v in cons_wrong_row.values() if v is not None]) / len([v for v in cons_wrong_row.values() if v is not None])
    
    all_results.append({
        'name': f"{os.path.basename(base_path)}",
        'acc': acc_row,
        'bias': bias_row,
        'cons_true': cons_true_row,
        'cons_wrong': cons_wrong_row
    })


all_task_types = set()
for res in all_results:
    all_task_types.update(res['acc'].keys())
all_task_types = sorted(all_task_types)
import csv

def save_csv(title, key, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["模型/任务"] + list(all_task_types)
        writer.writerow(header)
        for res in all_results:
            row = [res['name']]
            for task in all_task_types:
                val = res[key].get(task)
                if val is None:
                    row.append("-")
                else:
                    if key == 'bias':
                        row.append(f"{val:.3f}")
                    else:
                        row.append(f"{val:.4f}")
            writer.writerow(row)
    print(f"{title} 已保存为 {filename}")

save_csv("准确率（Accuracy）", "acc", "acc.csv")
save_csv("Yes/No Bias Score", "bias", "bias.csv")
save_csv("Consistency True", "cons_true", "consistency_true.csv")
save_csv("Consistency Wrong", "cons_wrong", "consistency_wrong.csv")

