import random
import re
import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from ..utils.misc import print_once
from .base import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

Semantic = ['homophone','inferredsound','knowledge','overreliance','polysemy','prosodic']

class Qwen2Audio(BaseModel):
    NAME = 'Qwen2-Audio-7B'

    def __init__(self, model_path='Qwen/Qwen2-Audio-7B', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(
            model_path
        )

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path, device_map='cuda').eval()
        random.seed(0)
        torch.cuda.empty_cache()

    def get_prompt(self, msg: dict):
        meta = msg['meta']
        print(meta)
        print(meta['type'])
        if 'asr' in meta['type']:
            lang = meta['type'].split('_')[-1]
            prompt = f'Detect the language and recognize the speech: <|{lang}|>'
        elif any(a in meta['type'] for a in Semantic):
            prompt = msg['text'] + ' The answer is:'
        else:
            prompt = prompt = f'Listen to the given audio carefully and answer this question: {msg["text"]}.'
        # if meta['task'] == 'ASR':
        #     assert 'lang' in meta
        #     lang = meta['lang']
        #     # from jsonl in: https://github.com/QwenLM/Qwen2-Audio/blob/main/eval_audio/EVALUATION.md
        #     prompt = f'Detect the language and recognize the speech: <|{lang}|>'
        # elif meta['dataset_name'] == 'meld':
        #     # from: https://github.com/QwenLM/Qwen2-Audio/blob/main/eval_audio/EVALUATION.md
        #     prompt = 'Recognize the emotion with keywords in English:'
        # elif meta['dataset_name'] == 'vocalsound':
        #     # from: https://github.com/QwenLM/Qwen2-Audio/blob/main/eval_audio/EVALUATION.md
        #     prompt = 'Classify the human vocal sound to VocalSound in English:'
        # # help to invoke baesmodel continuous output
        # elif meta['interactive'] == 'Audio-QA':
        #     prompt = ' Your answer to the question is:'
        # elif meta['audio_type'] == 'AudioEvent':
        #     prompt = f'Listen to the given audio carefully and answer this question: {msg["text"]} Your answer is:'
        # else:
        #     prompt = msg['text'] + ' The answer is:'

        return '<|audio_bos|><|AUDIO|><|audio_eos|>' + prompt

    # 该模型是评测主力，只有chat才用chat模型
    def generate_inner(self, msg: dict):
        audio = None
        # 从message中提取audio和text
        audio = msg['audio']
        if len(audio) == 1:
            audio = audio[0]
        prompt = self.get_prompt(msg)

        print_once(f'Prompt: {prompt}')
        audio = librosa.load(
            audio, sr=self.processor.feature_extractor.sampling_rate)[0]

        inputs = self.processor(
            text=prompt,
            audios=audio,
            return_tensors='pt',
            sampling_rate=self.processor.feature_extractor.sampling_rate,
        )
        inputs = inputs.to('cuda')
        generated_ids = self.model.generate(**inputs, max_new_tokens=256, min_new_tokens=1, do_sample=False,
                                            top_k=None,
                                            top_p=None)
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        pred = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return prompt, pred









class Qwen2AudioChat(BaseModel):
    NAME = 'Qwen2-Audio-7B-Instruct'

    def __init__(self, model_path='Qwen/Qwen2-Audio-7B-Instruct', **kwargs):
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path, device_map='cuda'
        )
        torch.cuda.empty_cache()

    def get_prompt(self, msg: dict):
        meta = msg['meta']
        if 'asr' in meta['type'] or any(a in meta['type'] for a in Semantic):
            prompt = msg["text"]
        else:
            prompt = f'Listen to the given audio carefully and answer this question: {msg["text"]}.'
        # if meta['audio_type'] == 'AudioEvent':
        #     prompt = f'Listen to the given audio carefully and answer this question: {msg["text"]}.'
        # else:
        #     prompt = msg['text']
        return prompt

    def generate_inner(self, msg: dict):
        audio = msg['audio']
        if len(audio) == 1:
            audio = audio[0]

        prompt = ''
        # if msg['meta']['interactive'] == 'Audio-QA':
        #     conversation = [{'role': 'user',
        #                      'content': [{'type': 'audio',
        #                                   'audio_url': audio}]}]
        # else:
        prompt = self.get_prompt(msg)
        # from: https://github.com/QwenLM/Qwen2-Audio/blob/dfc7d31b0a3181c8be496155bbf9eb3049499b3c/README.md?plain=1#L134
        conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user',
                            'content': [{'type': 'audio', 'audio_url': audio},
                                        {'type': 'text', 'text': prompt}]}]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios = []
        for message in conversation:
            if isinstance(message['content'], list):
                for ele in message['content']:
                    if ele['type'] == 'audio':
                        audios.append(
                            librosa.load(
                                ele['audio_url'],
                                sr=self.processor.feature_extractor.sampling_rate,
                            )[0]
                        )
        inputs = self.processor(
            text=text,
            audios=audios,
            return_tensors='pt',
            padding=True,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
        )
        inputs = inputs.to('cuda')
        generate_ids = self.model.generate(**inputs, max_new_tokens=256)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        answer = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        meta = msg['meta']
        if 'asr' in meta['type']:
            answer1 = re.findall(r"'([^']*)'", answer)
            if answer1:
                answer = answer1[0]

        return prompt, answer


class QwenAudio(BaseModel):
    NAME = 'Qwen-Audio'

    def __init__(self, model_path='Qwen/Qwen-Audio', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()
        random.seed(0)
        torch.cuda.empty_cache()

    def get_prompt(self, msg: dict):
        meta = msg['meta']
        print(meta)
        print(meta['type'])
        lang = meta['type'].split('_')[-1]
        if 'asr' in meta['type']:
            prompt = f'<|startoftranscript|><|{lang}|><|transcribe|><|{lang}|><|notimestamps|><|wo_itn|>'
        # elif any(a in meta['type'] for a in Semantic):
        #     prompt = msg['text'] + ' The answer is:'
        else:
            lang = meta['lang']
            prompt = f'<|startofanalysis|><|{lang}|><|question|>{msg["text"]}<|answer|>'
        return prompt

    # 该模型是评测主力，只有chat才用chat模型
    def generate_inner(self, msg: dict):
        audio = None
        
        prompt = self.get_prompt(msg)

        audio_url = msg['meta']['audio_path']
        query = f'<audio>{audio_url}</audio>' + prompt
        audio_info = self.tokenizer.process_audio(query)
        inputs = self.tokenizer(query, return_tensors='pt', audio_info=audio_info)
        inputs = inputs.to(self.model.device)
        pred = self.model.generate(**inputs, audio_info=audio_info)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True,audio_info=audio_info)
        response = response.split(msg["text"])[-1].split('.wav')[-1]
        
        return prompt, response
