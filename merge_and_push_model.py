from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from config import *

best_checkpoint_path = os.path.join(FINE_TUNED_MODELS_PATH, BASE_MODEL_NAME + "_checkpoints", "checkpoint-4200")
merged_model_path = os.path.join(FINE_TUNED_MODELS_PATH, MERGED_MODEL_NAME)

if os.path.exists(merged_model_path):
    print(f"Load previous merged model from {merged_model_path}.")
    model_to_push = AutoModelForCausalLM.from_pretrained(merged_model_path, torch_dtype="auto", device_map="cuda")
else:
    print(f"Using {best_checkpoint_path} to merge model.")
    base_with_adapters_model = AutoPeftModelForCausalLM.from_pretrained(best_checkpoint_path, torch_dtype="auto", device_map=DEVICE)
    model_to_push = base_with_adapters_model.merge_and_unload()  
    model_to_push.save_pretrained(merged_model_path)  
    base_model_path = os.path.join(BASE_MODELS_PATH, BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(merged_model_path)  
