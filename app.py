# app.py (完整修正后的代码)

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch  # 导入 torch，用于设备管理
import json   # 导入 json，用于加载现有广告评论数据库

# 导入评论评估器模块中的函数
from comment_evaluator import (
    evaluate_comment_full_pipeline,
    load_ad_classifier_model,
    load_sbert_model,
    load_sentiment_model
)

app = Flask(__name__)
# 允许所有来源的跨域请求。在生产环境中，你可能希望限制为特定的前端域名。
CORS(app)

# 全局变量来存储预加载的模型和数据库
# 这样模型只会在应用启动时加载一次，后续请求直接使用这些实例
AD_CLASSIFIER_MODEL = None
AD_CLASSIFIER_TOKENIZER = None
SBERT_MODEL = None
SENTIMENT_PIPELINE = None
EXISTING_AD_COMMENTS_DB = [] # 初始化为空列表，等待从文件加载

# 在第一个请求之前加载所有模型和数据
# @app.before_request 确保在处理任何请求之前执行
# Gunicorn 通常会为每个 worker 进程运行一次这个初始化
@app.before_request
def initialize_models():
    global AD_CLASSIFIER_MODEL, AD_CLASSIFIER_TOKENIZER, SBERT_MODEL, SENTIMENT_PIPELINE, EXISTING_AD_COMMENTS_DB

    # 只有当模型尚未加载时才执行加载过程
    if AD_CLASSIFIER_MODEL is None: # 以广告分类模型作为检查点
        print("--- Flask App Initialization: Preloading models ---")

        # 确保cuda可用，否则使用cpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for model loading and inference: {device}")

        # 1. 加载广告分类模型
        print("Loading ad classifier model...")
        AD_CLASSIFIER_TOKENIZER, AD_CLASSIFIER_MODEL = load_ad_classifier_model(device)
        print(f"DEBUG (app.py): AD_CLASSIFIER_MODEL type: {type(AD_CLASSIFIER_MODEL)}")
        if AD_CLASSIFIER_MODEL:
            print(f"DEBUG (app.py): AD_CLASSIFIER_MODEL device: {AD_CLASSIFIER_MODEL.device}")
        print("Ad classifier model and tokenizer loaded successfully.")

        # 2. 加载 Sentence-BERT 模型
        print("Loading Sentence-BERT model: paraphrase-multilingual-MiniLM-L12-v2...")
        SBERT_MODEL = load_sbert_model(device)
        print("Sentence-BERT model loaded successfully.")

        # 3. 加载情感分析模型
        print("Loading sentiment model: IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment...")
        SENTIMENT_PIPELINE = load_sentiment_model(device)
        print("Sentiment model loaded successfully.")

        # 4. 加载广告评论数据库
        # 假设你的 existing_ad_comments.json 在应用程序的当前工作目录下
        try:
            with open('existing_ad_comments.json', 'r', encoding='utf-8') as f:
                EXISTING_AD_COMMENTS_DB = json.load(f)
            print("Existing ad comments database loaded successfully.")
        except FileNotFoundError:
            print("Warning: existing_ad_comments.json not found. Modified spam detection will not work.")
            EXISTING_AD_COMMENTS_DB = [] # 如果文件不存在，设置为空列表
        except json.JSONDecodeError:
            print("Error: existing_ad_comments.json is not valid JSON. Modified spam detection will not work.")
            EXISTING_AD_COMMENTS_DB = []
        except Exception as e:
            print(f"Error loading existing_ad_comments.json: {e}")
            EXISTING_AD_COMMENTS_DB = []

        print("All models and data preloaded successfully for Flask app.")


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({"status": "healthy", "message": "Comment evaluation service is up and running."})

@app.route('/evaluate_comment', methods=['POST'])
def evaluate_comment():
    """
    API 接口，用于接收单条评论文本，并返回其审核结果。
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    comment_text = data.get('comment')

    if not comment_text or not isinstance(comment_text, str):
        return jsonify({"error": "Invalid input: 'comment' field is missing or not a string."}), 400

    print(f"Received single comment for evaluation: '{comment_text}'")

    try:
        # 调用 comment_evaluator 模块中的主评估函数，并传入预加载的模型实例
        print(f"DEBUG (app.py - single): AD_CLASSIFIER_MODEL device before call: {AD_CLASSIFIER_MODEL.device}")
        result = evaluate_comment_full_pipeline(
            comment_text,
            EXISTING_AD_COMMENTS_DB,
            AD_CLASSIFIER_MODEL,
            AD_CLASSIFIER_TOKENIZER,
            SBERT_MODEL,
            SENTIMENT_PIPELINE
        )
        return jsonify(result), 200
    except Exception as e:
        print(f"Error processing comment '{comment_text}': {e}")
        return jsonify({"error": f"Internal server error during evaluation: {str(e)}"}), 500

@app.route('/evaluate_comments_batch', methods=['POST']) # <--- 新增的批量处理API路由
def evaluate_comments_batch():
    """
    API 接口，用于接收评论文本列表，并返回所有评论的审核结果列表。
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    comments = data.get('comments') # <--- 期望接收一个名为 'comments' 的列表

    if not isinstance(comments, list):
        return jsonify({"error": "Invalid input: 'comments' must be a list of strings."}), 400

    if not comments: # 如果列表为空，直接返回空结果
        return jsonify({"results": []}), 200

    all_results = []
    for comment_text in comments:
        if not isinstance(comment_text, str):
            all_results.append({"comment": comment_text, "error": "Comment item is not a string."})
            continue # 跳过当前非字符串项，继续处理下一条

        try:
            # 确保传递预加载的模型和数据库
            print(f"DEBUG (app.py - batch): AD_CLASSIFIER_MODEL device before call: {AD_CLASSIFIER_MODEL.device}")
            result = evaluate_comment_full_pipeline(
                comment_text,
                EXISTING_AD_COMMENTS_DB,
                AD_CLASSIFIER_MODEL,
                AD_CLASSIFIER_TOKENIZER,
                SBERT_MODEL,
                SENTIMENT_PIPELINE
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error processing comment '{comment_text}': {e}")
            all_results.append({"comment": comment_text, "error": f"Error processing comment: {str(e)}"})

    return jsonify({"results": all_results}), 200

if __name__ == '__main__':
    # 监听所有可用的IP地址，端口5000
    # 在生产环境中，请使用Gunicorn等WSGI服务器来运行Flask应用
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=True 适合开发，生产环境应设为False