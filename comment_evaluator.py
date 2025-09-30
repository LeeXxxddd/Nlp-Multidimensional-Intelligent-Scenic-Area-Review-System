# comment_evaluator.py (完整修正后的代码)

import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 确保使用国内镜像加速Hugging Face模型下载
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, util
import Levenshtein  # 用于编辑距离计算
import numpy as np

# --- 常量 ---
# 注意：AD_CLASSIFIER_MODEL_PATH 应该指向你训练好的模型路径
AD_CLASSIFIER_MODEL_PATH = "/root/autodl-tmp/lee/results/ad_classifier_model/checkpoint-1500"
AD_LABEL_MAP = {0: '非广告', 1: '广告'}  # 确保这里的标签与你的模型输出对应
SENTENCE_BERT_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
SENTIMENT_MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment"


# --- 模型加载函数 (这些函数将从 app.py 调用一次，并返回模型实例) ---

def load_ad_classifier_model(device):
    """
    加载广告评论分类模型和分词器。
    Args:
        device (torch.device): 指定模型加载到的设备 (cuda 或 cpu)。
    Returns:
        tuple: (tokenizer, model)
    """
    print("Loading ad classifier model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(AD_CLASSIFIER_MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(AD_CLASSIFIER_MODEL_PATH)
        model.to(device)
        model.eval()  # 设置为评估模式
        print(f"DEBUG (evaluator): Loaded ad classifier model device: {model.device}")
        print("Ad classifier model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading ad classifier model from {AD_CLASSIFIER_MODEL_PATH}: {e}")
        raise RuntimeError(f"Failed to load ad classifier model: {e}")


def load_sbert_model(device):
    """
    加载 Sentence-BERT 模型。
    Args:
        device (torch.device): 指定模型加载到的设备 (cuda 或 cpu)。
    Returns:
        SentenceTransformer: 加载后的 Sentence-BERT 模型实例。
    """
    print(f"Loading Sentence-BERT model: {SENTENCE_BERT_MODEL_NAME}...")
    try:
        # SentenceTransformer 的 device 参数可以直接接受 torch.device 对象
        model = SentenceTransformer(SENTENCE_BERT_MODEL_NAME, device=device)
        print(f"Sentence-BERT model loaded successfully on {model.device}.")
        return model
    except Exception as e:
        print(f"Error loading Sentence-BERT model {SENTENCE_BERT_MODEL_NAME}: {e}")
        raise RuntimeError(f"Failed to load Sentence-BERT model: {e}")


def load_sentiment_model(device):
    """
    加载情感分析模型。
    Args:
        device (torch.device): 指定模型加载到的设备 (cuda 或 cpu)。
    Returns:
        transformers.pipelines.Pipeline: 加载后的情感分析 pipeline 实例。
    """
    print(f"Loading sentiment model: {SENTIMENT_MODEL_NAME}...")
    # pipeline 的 device 参数可以接受整数（GPU ID）或字符串（"cuda:0", "cpu"）
    # 这里直接使用传入的 device 对象的 index 或 type
    pipeline_device = device.index if device.type == 'cuda' else -1
    try:
        sentiment_pipeline_instance = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME, device=pipeline_device)
        print("Sentiment model loaded successfully.")
        return sentiment_pipeline_instance
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        raise RuntimeError(f"Failed to load sentiment model: {e}")


# --- 评论分类函数 (接受模型作为参数) ---
def classify_comment_as_ad(comment_text: str, ad_model, ad_tokenizer, ad_device) -> dict:
    """
    对给定的评论文本进行广告分类。
    Args:
        comment_text (str): 待分类的评论文本。
        ad_model: 已加载的广告分类模型实例。
        ad_tokenizer: 已加载的广告分类分词器实例。
        ad_device: 模型所在的设备 (用于将输入张量移到正确的设备)。
    Returns:
        dict: 包含分类标签和概率。
    """
    print(f"DEBUG (evaluator): classify_comment_as_ad called for '{comment_text}'")
    print(f"DEBUG (evaluator): ad_model device received: {ad_model.device}")
    print(f"DEBUG (evaluator): ad_device received (for input tensor): {ad_device}")

    inputs = ad_tokenizer(comment_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # 将分词器输出的张量移动到正确的设备上
    inputs = {k: v.to(ad_device) for k, v in inputs.items()}

    ad_model.eval()  # 再次确保模型处于评估模式
    with torch.no_grad():
        outputs = ad_model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class_id = torch.argmax(probabilities, dim=1).item()

    predicted_label = AD_LABEL_MAP[predicted_class_id]

    # 返回广告（1）的概率，确保对应正确的索引
    # 如果 AD_LABEL_MAP = {0: '非广告', 1: '广告'}，那么索引 1 对应 '广告'
    ad_probability = float(probabilities[0, 1])

    return {
        "label": predicted_label,
        "probabilities": probabilities[0].tolist(),  # 返回所有标签的概率列表
        "ad_probability": ad_probability  # 单独返回广告的概率，方便后续展示
    }


# --- 情感分析函数 (接受 pipeline 作为参数) ---
def analyze_sentiment(text: str, sentiment_pipeline) -> dict:
    """
    分析评论的情感倾向，返回情感标签和分数。
    Args:
        text (str): 待分析的评论文本。
        sentiment_pipeline: 已加载的情感分析 pipeline 实例。
    Returns:
        dict: 包含情感标签和分数。
    """
    if sentiment_pipeline:
        try:
            result = sentiment_pipeline(text)[0]
            # 根据模型输出的标签进行映射
            # 'positive', 'negative', 'neutral' 都是小写
            label_map = {'positive': '积极', 'negative': '消极', 'neutral': '中性'}
            return {
                "sentiment_label": label_map.get(result['label'].lower(), result['label']),
                "sentiment_score": float(result['score'])
            }
        except Exception as e:
            print(f"Error during sentiment analysis for '{text}': {e}")
            return {"sentiment_label": "分析失败", "sentiment_score": 0.0, "error": str(e)}
    return {"sentiment_label": "模型未加载", "sentiment_score": 0.0, "error": "Sentiment pipeline not loaded."}


# --- 计算语义相似度 (接受 sbert_model 作为参数) ---
def get_sentence_embedding(text: str, sbert_model):
    """
    获取文本的 Sentence-BERT 嵌入。
    Args:
        text (str): 待嵌入的文本。
        sbert_model: 已加载的 Sentence-BERT 模型实例。
    Returns:
        torch.Tensor or None: 文本嵌入。
    """
    if sbert_model is None:
        print("Sentence-BERT model not available for embedding.")
        return None
    sbert_model.eval()  # 设置为评估模式
    with torch.no_grad():
        # 确保编码时使用模型所在的设备
        embedding = sbert_model.encode(text, convert_to_tensor=True, device=sbert_model.device)
    return embedding


def calculate_semantic_similarity(text1: str, text2: str, sbert_model) -> float:
    """
    计算两个文本之间的余弦相似度。
    Args:
        text1 (str): 第一个文本。
        text2 (str): 第二个文本。
        sbert_model: 已加载的 Sentence-BERT 模型实例。
    Returns:
        float: 余弦相似度。
    """
    embedding1 = get_sentence_embedding(text1, sbert_model)
    embedding2 = get_sentence_embedding(text2, sbert_model)
    if embedding1 is None or embedding2 is None:
        return 0.0
    # 余弦相似度计算，确保嵌入在同一个设备上
    cosine_similarity = util.pytorch_cos_sim(embedding1.to(sbert_model.device),
                                             embedding2.to(sbert_model.device)).item()
    return cosine_similarity


# --- 计算编辑距离相似度 ---
def calculate_edit_distance_similarity(text1: str, text2: str) -> float:
    """
    计算两个文本之间的编辑距离相似度。
    """
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0

    edit_dist = Levenshtein.distance(text1, text2)
    max_len = max(len(text1), len(text2))
    if max_len == 0:  # 避免除以零
        return 1.0
    similarity = 1 - (edit_dist / max_len)
    return similarity


# --- 复制修改评论识别模块 (接受 sbert_model 作为参数) ---
def identify_modified_spam(new_comment: str, existing_ad_comments: list[str], sbert_model,
                           semantic_threshold: float = 0.75, edit_distance_threshold: float = 0.7) -> dict:
    """
    识别一条新评论是否是现有广告评论的复制修改版本。
    Args:
        new_comment (str): 待识别的新评论。
        existing_ad_comments (list[str]): 已知为广告的评论列表。
        sbert_model: 已加载的 Sentence-BERT 模型实例。
        semantic_threshold (float): 语义相似度阈值。
        edit_distance_threshold (float): 编辑距离相似度阈值。
    Returns:
        dict: 识别结果。
    """
    if not existing_ad_comments:
        return {"is_modified_spam": False, "matched_original": None, "semantic_similarity": 0.0,
                "edit_distance_similarity": 0.0, "reason": "No existing ad comments to compare against."}

    # 确保 S-BERT 模型可用
    if sbert_model is None:
        return {"is_modified_spam": False, "matched_original": None, "semantic_similarity": 0.0,
                "edit_distance_similarity": 0.0, "reason": "S-BERT model not loaded."}

    max_semantic_similarity = 0
    max_edit_distance_similarity = 0
    matched_original_comment = ""

    # 获取新评论的嵌入一次
    new_comment_embedding = get_sentence_embedding(new_comment, sbert_model)
    if new_comment_embedding is None:
        return {"is_modified_spam": False, "matched_original": None, "semantic_similarity": 0.0,
                "edit_distance_similarity": 0.0, "reason": "Failed to get embedding for new comment."}

    for original_comment in existing_ad_comments:
        original_embedding = get_sentence_embedding(original_comment, sbert_model)
        if original_embedding is None:
            continue  # 跳过无法获取嵌入的原始评论

        semantic_similarity = util.pytorch_cos_sim(new_comment_embedding, original_embedding).item()
        edit_distance_similarity = calculate_edit_distance_similarity(new_comment, original_comment)

        if semantic_similarity > max_semantic_similarity:
            max_semantic_similarity = semantic_similarity
            matched_original_comment = original_comment

        if edit_distance_similarity > max_edit_distance_similarity:
            max_edit_distance_similarity = edit_distance_similarity

    # 定义相似度阈值 (根据你之前的设定，0.01 的编辑距离阈值非常低)
    SEMANTIC_THRESHOLD = 0.75
    EDIT_DISTANCE_THRESHOLD = 0.01  # 你的原设置为 0.01，如果想更严格，可提高，例如 0.5 或 0.7

    is_modified_spam = False
    reason = "No match found."

    if max_semantic_similarity >= SEMANTIC_THRESHOLD and max_edit_distance_similarity >= EDIT_DISTANCE_THRESHOLD:
        is_modified_spam = True
        reason = "High semantic and edit distance similarity to an existing ad comment."
    elif max_semantic_similarity >= SEMANTIC_THRESHOLD and len(
            new_comment) < 30 and max_semantic_similarity > 0.9:  # 针对短评论高语义相似度
        is_modified_spam = True
        reason = "High semantic similarity for a short comment, potentially a modified ad."
    else:
        # 提供更详细的非修改垃圾评论原因
        if max_semantic_similarity < SEMANTIC_THRESHOLD:
            reason = f"Semantic similarity ({max_semantic_similarity:.4f}) below threshold ({SEMANTIC_THRESHOLD})."
        elif max_edit_distance_similarity < EDIT_DISTANCE_THRESHOLD:
            reason = f"Edit distance similarity ({max_edit_distance_similarity:.4f}) below threshold ({EDIT_DISTANCE_THRESHOLD})."
        else:
            reason = "No match found."  # 最终默认原因

    return {
        "is_modified_spam": is_modified_spam,
        "matched_original": matched_original_comment if is_modified_spam else "N/A",
        "semantic_similarity": max_semantic_similarity,
        "edit_distance_similarity": max_edit_distance_similarity,
        "reason": reason
    }


# --- 完整评论评估管道函数 (现在接受所有模型实例作为参数) ---
def evaluate_comment_full_pipeline(comment_text: str, existing_ad_comments: list[str],
                                   ad_classifier_model, ad_classifier_tokenizer,
                                   sbert_model, sentiment_pipeline) -> dict:
    """
    对评论进行完整的审核流程：先进行广告分类，如果是广告，再检测是否为复制修改的垃圾评论，并进行情感分析。

    Args:
        comment_text (str): 待审核的评论文本。
        existing_ad_comments (list[str]): 已知为广告的评论列表。
        ad_classifier_model: 广告分类模型实例。
        ad_classifier_tokenizer: 广告分类分词器实例。
        sbert_model: Sentence-BERT 模型实例。
        sentiment_pipeline: 情感分析 pipeline 实例。

    Returns:
        dict: 包含审核结果的字典。
              - 'comment': str, 评论原文
              - 'is_ad': bool, 是否被广告分类模型识别为广告。
              - 'classification_label': str, 广告分类模型的预测标签 ('广告'/'非广告')。
              - 'ad_probability': float, 广告分类模型的广告概率。
              - 'is_modified_spam': bool, 是否被识别为复制修改的垃圾评论。
              - 'modified_spam_details': dict, 复制修改检测的详细结果 (包含匹配到的原始评论、相似度等)。
              - 'sentiment': dict, 情感分析结果 (包含 'sentiment_label' 和 'sentiment_score')。
    """
    print(f"Processing comment in pipeline: '{comment_text}'")

    # 1. 广告分类初步判断
    # 传递 ad_classifier_model.device 确保输入张量在正确的设备上
    ad_classification_result = classify_comment_as_ad(
        comment_text, ad_classifier_model, ad_classifier_tokenizer, ad_classifier_model.device
    )

    is_ad = (ad_classification_result["label"] == '广告')

    # 2. 如果初步判断是广告，则进一步检测是否为复制修改的垃圾评论
    # 也可以选择不区分是否广告都检测，取决于业务需求。
    modified_spam_details = {"is_modified_spam": False, "matched_original": None,
                             "semantic_similarity": 0.0, "edit_distance_similarity": 0.0,
                             "reason": "Not classified as ad or no modified spam match."}

    if is_ad:  # 只有当被初步识别为广告时，才进行详细的修改垃圾评论检测
        modified_spam_details = identify_modified_spam(
            new_comment=comment_text,
            existing_ad_comments=existing_ad_comments,
            sbert_model=sbert_model,  # 传递 S-BERT 模型
            semantic_threshold=0.75,
            edit_distance_threshold=0.01
        )

    # 3. 进行情感分析
    sentiment_result = analyze_sentiment(comment_text, sentiment_pipeline)  # 传递情感分析 pipeline

    return {
        "comment": comment_text,  # 返回评论原文，方便前端展示
        "is_ad": is_ad,
        "classification_label": ad_classification_result["label"],
        "ad_probability": ad_classification_result["ad_probability"],
        "is_modified_spam": modified_spam_details["is_modified_spam"],
        "modified_spam_details": modified_spam_details,
        "sentiment": sentiment_result
    }


# --- 示例使用 (此部分仅用于独立测试 comment_evaluator.py 模块，不影响 Flask 应用) ---
# 在 Flask 应用中，模型由 app.py 统一加载和管理，不会执行此处的 if __name__ == "__main__": 块

if __name__ == "__main__":
    print("--- Initializing models for comment_evaluator.py module testing ---")

    # 这里的设备管理是为本模块的独立测试准备的
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for module testing: {test_device}")

    # 加载所有模型实例
    try:
        test_ad_tokenizer, test_ad_model = load_ad_classifier_model(test_device)
        test_sbert_model = load_sbert_model(test_device)
        test_sentiment_pipeline = load_sentiment_model(test_device)
    except RuntimeError as e:
        print(f"模块测试模型加载失败: {e}")
        exit()

    # 模拟一个数据库中已知的广告评论列表
    existing_ad_comments_db_for_test = [
        "专业代写各类论文，价格合理，保质保量，详情加扣扣：123456789。",
        "想学编程？报名我们的Python入门班，名师一对一指导，保你学会！",
        "最新款手机低价出售，联系微信：test12345。",
        "提供高品质旅游服务，加微信号：travelpro。",
        "这家酒店服务真的太棒了，强烈推荐！联系电话13812345678。",
        "我发现一个超棒的购物平台，点击这里链接：www.example.com"
    ]

    comments_to_evaluate = [
        # 明显广告
        '这家酒店服务真的太棒了，强烈推荐！联系电话13812345678。',
        '专业代写各类论文，价格合理，保质保量，详情加扣扣：123456789。',

        # 非广告
        '景区很美，空气清新，值得一去。',
        '今天天气很好，玩得很开心。',
        '这是一个完全不同的评论，没有任何广告性质。',

        # 复制修改的垃圾评论测试
        "专业代写论文，合理价，包通过，Q号：123456789。",
        "想学编程？参加我们的Python基础课程，专业老师指导，保证学会！",
        "代写论文，加扣扣：123456789。",
        "提供优质旅游服务，加薇信：travelpro。",
        "酒店服务好，电话13812345678。",
        "发现超棒购物平台，点此链接：www.example.com"
    ]

    print("\n--- Running Full Comment Evaluation Pipeline with test models ---")
    for i, comment in enumerate(comments_to_evaluate):
        print(f"\n--- Processing Comment {i + 1} ---")
        print(f"评论: '{comment}'")

        full_result = evaluate_comment_full_pipeline(
            comment,
            existing_ad_comments_db_for_test,
            test_ad_model,
            test_ad_tokenizer,
            test_sbert_model,
            test_sentiment_pipeline
        )

        if "error" in full_result:
            print(f"处理出错: {full_result['error']}")
            continue

        print(f"  评论原文: {full_result['comment']}")
        print(f"  广告分类结果: {full_result['classification_label']} (广告概率: {full_result['ad_probability']:.4f})")
        print(f"  是否被识别为广告: {full_result['is_ad']}")
        print(f"  是否为复制修改垃圾评论: {full_result['is_modified_spam']}")

        if full_result['is_modified_spam']:
            details = full_result['modified_spam_details']
            print(f"    匹配到原始广告: '{details['matched_original']}'")
            print(f"    语义相似度: {details['semantic_similarity']:.4f}")
            print(f"    编辑距离相似度: {details['edit_distance_similarity']:.4f}")
            print(f"    原因: {details['reason']}")
        else:
            print(f"    复制修改检测原因: {full_result['modified_spam_details']['reason']}")

        sentiment_info = full_result.get('sentiment', {})
        print(
            f"  情感分析结果: 标签='{sentiment_info.get('sentiment_label', '未知')}', 分数={sentiment_info.get('sentiment_score', 0.0):.4f}")
        print("=" * 60)