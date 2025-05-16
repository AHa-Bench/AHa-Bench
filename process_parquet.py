import os
import json
import hashlib
import pandas as pd


type2id = {
    'source_number': 0,
    'distance': 10,
    'duration': 20,
    'repetition': 30,
    'inferred sound': 40,
    'temporal sequence': 50,
    'existence': 60,
    'Authenticity': 70,
    'homophone_en-en': 80,
    'homophone_zh-en': 90,
    'homophone_zh-zh': 100,
    'inferredsound_en': 110,
    'inferredsound_zh': 120,
    'knowledge_en': 130,
    'knowledge_zh': 140,
    'overreliance_en': 150,
    'overreliance_zh': 160,
    'polysemy_en': 170,
    'polysemy_zh': 180,
    'prosodic_en': 190,
    'prosodic_zh': 200,
    'asr_zh': 210,
    'asr_en': 220
}
# parquet path
file_path = ""
df = pd.read_parquet(file_path)

audio_col = 'audio'
type_col = 'type'
qid_col = 'question_id'

output_dir = 'kimi/dataset/aha_bench'
os.makedirs(output_dir, exist_ok=True)

results = []

type_audio_hash_dict = {}  # {type: {md5: (type_audio_id, wav_path)}}
type_audio_id_counter = {} # {type: 当前type下音频编号}
type_audio_id_to_count = {} # {type: {type_audio_id: 当前音频下的顺序号}}


for idx, row in df.iterrows():
    audio_data = row[audio_col]
    cur_type = str(row[type_col])
    

    wav_bytes = audio_data['bytes']
    md5 = hashlib.md5(wav_bytes).hexdigest()

    # 初始化
    if cur_type not in type_audio_hash_dict:

        type_audio_hash_dict[cur_type] = {}
        type_audio_id_counter[cur_type] = 0
        type_audio_id_to_count[cur_type] = {}

    audio_hash_dict = type_audio_hash_dict[cur_type]
    audio_id_counter = type_audio_id_counter[cur_type]
    audio_id_to_count = type_audio_id_to_count[cur_type]

    if md5 not in audio_hash_dict:
        type_audio_id = audio_id_counter
        type_audio_id_counter[cur_type] += 1
        save_dir = os.path.join(output_dir, cur_type)
        os.makedirs(save_dir, exist_ok=True)
        wav_path = os.path.join(save_dir, f"{type_audio_id}.wav")
        with open(wav_path, 'wb') as f:
            f.write(wav_bytes)
        audio_hash_dict[md5] = (type_audio_id, wav_path)
        audio_id_to_count[type_audio_id] = 1
    else:
        type_audio_id, wav_path = audio_hash_dict[md5]
        audio_id_to_count[type_audio_id] += 1

    cur_audio_count = audio_id_to_count[type_audio_id]
    new_qid = f"{type_audio_id}_{cur_audio_count}"

    record = row.to_dict()
    record['audio'] = ''
    record["audio_path"] = wav_path
    record["index"] = f"{type2id[cur_type]}_{new_qid}"
    record["audio_text"] = row['text']
    results.append(record)


jsonl_path = ""
with open(jsonl_path, "w", encoding="utf-8") as jf:
    for item in results:
        jf.write(json.dumps(item, ensure_ascii=False) + '\n')

