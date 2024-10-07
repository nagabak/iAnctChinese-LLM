import os
import json
import random
from config import DATA_PATH
import re

def remove_punctuation(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace(" ","")
    text=str(text)
    return text

def extract_pairs(file_path):
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):  
            if i + 1 < len(lines):  
                ancient_text = lines[i].strip().replace("古文：", "")
                modern_text = lines[i+1].strip().replace("现代文：", "")
                pair = {
                    "input": f"古文：{ancient_text} 现代文：",
                    "output": modern_text
                }
                pairs.append(pair)
                removed_text=remove_punctuation(ancient_text)
                pair = {
                    "input": f"无标点：{removed_text} 添加标点：",
                    "output": ancient_text
                }
                pairs.append(pair)
    return pairs

def recursive_search_and_extract(root_dir):
    all_pairs = []
    for root, dirs, files in os.walk(root_dir):
        if "bitext.txt" in files:
            file_path = os.path.join(root, "bitext.txt")
            pairs = extract_pairs(file_path)
            all_pairs.extend(pairs)
    return all_pairs

def split_data(pairs, test_ratio=0.2):
    random.shuffle(pairs)
    test_size = int(len(pairs) * test_ratio)
    test_set = pairs[:test_size]
    train_set = pairs[test_size:]
    return train_set, test_set

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    root_dir = os.path.join(DATA_PATH, "original")
    output_dir = os.path.join(DATA_PATH, "processed")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_pairs = recursive_search_and_extract(root_dir)
    
    train_set, val_set = split_data(all_pairs, test_ratio=0.2)
    
    save_json(train_set, os.path.join(output_dir, "train_data.json"))
    save_json(val_set, os.path.join(output_dir, "val_data.json"))

    print(f"数据处理完成：训练集 {len(train_set)} 条，验证集 {len(val_set)} 条。")
