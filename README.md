# 古汉语大模型

## 简介

输入古汉语，输出现代汉语；输入无标点的古汉语，输出有标点的古汉语。基于荀子基座大模型Xunzi-Qwen2-1.5B(https://github.com/Xunzi-LLM-of-Chinese-classics/XunziALLM?tab=readme-ov-file )，用“文言文（古文）- 现代文平行语料”(https://github.com/NiuTrans/Classical-Modern )数据微调训练得到。


## 使用说明

1. requirements.txt内为需要的库
2. data用于存储原始数据和处理好的数据，BaseModel存储基座模型
3. config.py基座模型路径和名称,微调后的模型路径和名称等基础数据
4. get_data.py，用于处理数据
5. finetune.py用于微调模型
6. merge_and_push_model.py用于将最佳checkpoint与基座模型融合得到微调好的模型，存储到finetuned_model文件夹
7. get_response.py用于调用调好的模型
8. main.py为用qt写的图形界面。
