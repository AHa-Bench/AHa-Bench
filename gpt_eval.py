import requests
import json
from tqdm import tqdm
import argparse
import os

prompt_template = (
    "You are an AI assistant who will help me to match an answer with two options of a question. "
    "The options are only Yes / No. "
    "You are provided with a question and an answer, "
    "and you need to find which option (Yes / No) is most similar to the answer. "
    "If the meaning of all options are significantly different from the answer, output Unknown. "
    "Your should output a single word among the following 3 choices: Yes, No, Unknown.\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
)

def call_llm(question, answer, model_name):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt_template.format(question=question, answer=answer)}
        ]
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    response.raise_for_status()
    content = response.json()['data']['choices'][0]['message']['content'].strip()
    return content

def process_jsonl_line(json_obj, model_name):
    question = json_obj.get('question', '')
    prediction = json_obj.get('prediction', '')
    prediction_result = call_llm(question, prediction, model_name)
    return prediction_result

def main(input_file, output_file, model_name):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, desc="Processing JSONL lines"):
            line = line.strip()
            if not line:
                continue
            while True:
                try:
                    json_obj = json.loads(line)
                    prediction_result = process_jsonl_line(json_obj, model_name)
                    json_obj['prediction_match'] = prediction_result
                    outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                    break  
                except Exception as e:
                    print(f"Error processing line: {e}，retrying...")
                    continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process JSONL with LLM model")
    parser.add_argument('--input_file', type=str, required=True, help='输入文件路径前缀')
    parser.add_argument('--output_file', type=str, required=True, help='输出文件路径前缀')
    parser.add_argument('--num', type=str, required=True, help='编号')
    parser.add_argument('--model_name', type=str, required=True, help='模型名称')
    args = parser.parse_args()

    input_file = f"{args.input_file}_{args.num}.jsonl"
    output_file = f"{args.output_file}_{args.num}.jsonl"
    model_name = args.model_name

    main(input_file, output_file, model_name)
