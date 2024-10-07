from datasets import load_dataset
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
from config import *
from datasets import Dataset
from datasets import load_from_disk

lora_config_dict = {
    "target_modules": ["q_proj", "k_proj", "v_proj"],
    "inference_mode": False,
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1
}
traning_args_dict = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 3,
    "logging_steps": 50,
    "save_steps": 600,
    "learning_rate": 1e-4,
    "do_eval": False,
    "gradient_checkpointing": True,
}

def process_func(sentence: dict) -> dict:

    input_ids, attention_mask, labels = [], [], []

    original_text = tokenizer(sentence['input'],
                                    add_special_tokens=False,
                                    max_length=max_source_length,
                                    truncation=True)
    translation = tokenizer(sentence['output'],
                                    add_special_tokens=False,
                                    max_length=max_target_length,
                                    truncation=True)

    input_ids = original_text["input_ids"] + translation["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = original_text["attention_mask"] + translation["attention_mask"] + [1]
    labels = [-100] * len(original_text["input_ids"]) + translation["input_ids"] + [tokenizer.pad_token_id]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def get_tokenized_dataset(dataset: Dataset, base_path: str, name: str) -> Dataset:
    tokenized_dataset_path = os.path.join(base_path, f'tokenized_{name}_data')
    if os.path.exists(tokenized_dataset_path):
        tokenized_dataset = load_from_disk(tokenized_dataset_path)
    else:
        tokenized_dataset = dataset.map(
            process_func,
            remove_columns=dataset.column_names,
        )

        tokenized_dataset.save_to_disk(tokenized_dataset_path)
    return tokenized_dataset

max_source_length = 256  
max_target_length = 128  
output_dir = os.path.join(FINE_TUNED_MODELS_PATH, BASE_MODEL_NAME + '_checkpoints')
swanlab_project_name = BASE_MODEL_NAME.replace(".", "_") + "_finetune_project"
swanlab_experiment_name = BASE_MODEL_NAME.replace(".", "_") + "_finetune_experiment3"

model_path = 'C:\\college\\dasanshang\\xt\\iAnctChinese-LLM\\BaseModel\\Xunzi-Qwen2-1.5B'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", torch_dtype="auto")
model.enable_input_require_grads() 

data_path = os.path.join(DATA_PATH, 'processed')
train_dataset = load_dataset("json", data_files = os.path.join(data_path, 'train_data.json'), split='train')
test_dataset = load_dataset("json", data_files = os.path.join(data_path, 'val_data.json'), split='train')  

tokenized_train_dataset = get_tokenized_dataset(train_dataset, data_path, 'train')


tokenized_validation_dataset = None
if traning_args_dict["do_eval"]:
    validation_dataset_path = os.path.join(data_path, 'validation_data.json')
    if os.path.exists(validation_dataset_path):
        validation_dataset = load_dataset("json", data_files = validation_dataset_path, split='train')
        tokenized_validation_dataset = get_tokenized_dataset(validation_dataset, data_path, 'validation')

lora_config_dict['task_type'] = TaskType.CAUSAL_LM
config = LoraConfig(**lora_config_dict)

model = get_peft_model(model, config)
model.print_trainable_parameters()

traning_args_dict["output_dir"] = output_dir
args = TrainingArguments(**traning_args_dict)


swanlab_callback = SwanLabCallback(project=swanlab_project_name, experiment_name=swanlab_experiment_name)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

#trainer.train(resume_from_checkpoint = True)
trainer.train()
print("微调完成！")

swanlab.finish()
