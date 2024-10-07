import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from config import *
import torch
fine_tuned_model_path = os.path.join(FINE_TUNED_MODELS_PATH, MERGED_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path, torch_dtype="auto", device_map=DEVICE)
model.generation_config.pad_token_id = tokenizer.pad_token_id

s_front=[ "古文：","无标点:"]
s_behind=["现代文：","添加标点:"]


def generate_answer(prompt, tokenizer, device, model):
    model_inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(device)
    generated_ids = model.generate(
        model_inputs['input_ids'],
        attention_mask=model_inputs['attention_mask'],
        max_new_tokens=128
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs['input_ids'], generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def split_and_generate(input_text,i):
    if input_text == '':
        return "请输入文字"
    sentences = re.findall(r'[^。！？]*[。！？]', input_text)
    if not sentences:
        sentences = [input_text]
    responses = ""
    for sentence in sentences:
        input = s_front[i] + sentence + s_behind[i] 
        response = generate_answer(input, tokenizer, DEVICE, model)
        responses += response
    return responses


print(split_and_generate("夫和戎狄君之幸也",1))