# model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897" # 设置环境变量，使用代理下载模型


def load_model(device=None):
    model_name = "EleutherAI/pythia-70m"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    model.eval()
    return model, tokenizer, device