# download_model.py

import os
from transformers import AutoTokenizer, AutoModel
import logging

# --- 配置 ---
# 设置日志格式，方便观察
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 指定要下载的模型名称 (Hugging Face Hub上的官方名称)
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"

# 指定模型要保存到的本地路径 (在项目根目录下的 model/ 文件夹内)
SAVE_DIRECTORY = os.path.join("model", "chinese-roberta-wwm-ext")

# --- 脚本主逻辑 ---
def download_model():
    """
    检查模型是否存在于本地，如果不存在，则从Hugging Face Hub下载。
    """
    logging.info(f"指定的模型名称: {MODEL_NAME}")
    logging.info(f"将要保存的路径: {SAVE_DIRECTORY}")

    # 检查目标文件夹是否已存在且不为空
    if os.path.exists(SAVE_DIRECTORY) and os.listdir(SAVE_DIRECTORY):
        logging.info("模型文件夹已存在且不为空，跳过下载。")
        return

    logging.info("模型未在本地找到，开始从 Hugging Face Hub 下载...")
    
    try:
        # 创建目标文件夹（如果不存在）
        os.makedirs(SAVE_DIRECTORY, exist_ok=True)

        # 下载分词器 (Tokenizer)
        logging.info("正在下载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(SAVE_DIRECTORY)
        logging.info("Tokenizer 下载并保存成功。")

        # 下载模型 (Model)
        logging.info("正在下载 Model... (这可能需要一些时间，请耐心等待)")
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.save_pretrained(SAVE_DIRECTORY)
        logging.info("Model 下载并保存成功。")

        logging.info("模型已成功下载到 '{}'".format(SAVE_DIRECTORY))

    except Exception as e:
        logging.error(f"下载过程中发生错误: {e}")
        logging.error("请检查网络连接或模型名称是否正确。")


if __name__ == "__main__":
    download_model()