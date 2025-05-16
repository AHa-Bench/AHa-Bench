# Code for AHa-bench evaluation

## abstract
Hallucinations present a significant challenge in the development and evaluation of large language models (LLMs), directly affecting their reliability and accuracy. While notable advancements have been made in research on textual and visual hallucinations, there is still a lack of a comprehensive benchmark for evaluating auditory hallucinations in large audio language models (LALMs). To fill this gap, we introduce \textbf{AHa-Bench}, a systematic and comprehensive benchmark for audio hallucinations. Audio data, in particular, uniquely combines the multi-attribute complexity of visual data with the semantic richness of textual data, leading to auditory hallucinations that share characteristics with both visual and textual hallucinations. Based on the source of these hallucinations, AHa-Bench categorizes them into semantic hallucinations, acoustic hallucinations, and semantic-acoustic confusion hallucinations. In addition, we systematically evaluate eight open-source local perception language models (LALMs), demonstrating the challenges these models face in audio understanding, especially when it comes to jointly understanding semantic and acoustic information. Through the development of a comprehensive evaluation framework, AHa-Bench aims to enhance the robustness and stability of LALMs, fostering more reliable and nuanced audio understanding in LALMs.


## Run the evaluation on AHa-bench
### Clone our repository
```
git clone https://github.com/AHa-Bench/AHa-Bench
cd ./AHa-Bench
```

### Install the requirements
```
pip install -r requirements.txt
```

### download and process the eval data 
```
#  https://huggingface.co/datasets/ahabench/AHa-Bench
python process_parquet.py
```

### Config the dataset_root
```yaml
DATASETS:
  dataset_root: "/path/to/your/dataset/root"
```

### Run the model inference on AHa-Bench
```
bash run_audio.sh --model {model} --data aha --skip-eval --force-reinfer --num{num}
```

### Run the GPT evaluation to match the answer
```
python gpt_eval.py --input_file {input_file} --output_file {output_file} --model_name {eval_gpt_model_name}
```

### Calculate the metrics
```
python eval_metric.py --input_file {input_file} --output_file {output_file}
```


## Add a new model
```
#todo: implement the generate_inner function
def generate_inner(self, msg:dict) -> (str, str)
```


## References
This implementation was developed based on the following repository:
* Kimi-Audio-Evalkit: <https://github.com/MoonshotAI/Kimi-Audio-Evalkit> (for architecture backbone)
