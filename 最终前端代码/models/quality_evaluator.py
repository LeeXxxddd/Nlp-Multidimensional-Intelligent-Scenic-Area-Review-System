import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import Levenshtein
import re
import jieba
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime

class ReviewQualityEvaluator:
    """高级评论质量评估器"""
    
    def __init__(self, model_name="uer/sbert-base-chinese-nli"):
        """初始化评估器"""
        self.sentence_model = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        
        # 质量评估权重
        self.weights = {
            'length_score': 0.15,
            'complexity_score': 0.20,
            'sentiment_score': 0.15,
            'information_score': 0.25,
            'originality_score': 0.25
        }
        
        # 初始化特征词典
        self._init_feature_dicts()
    
    def _init_feature_dicts(self):
        """初始化特征词典"""
        # 信息性词汇
        self.info_keywords = {
            '位置': ['位置', '地理', '交通', '方便', '到达', '停车'],
            '服务': ['服务', '态度', '热情', '专业', '礼貌', '周到'],
            '设施': ['设施', '环境', '卫生', '干净', '整洁', '现代'],
            '价格': ['价格', '费用', '性价比', '便宜', '贵', '值得'],
            '体验': ['体验', '感受', '印象', '满意', '推荐', '值得']
        }
        
        # 复杂性指标词汇
        self.complexity_indicators = [
            '因为', '所以', '但是', '然而', '不过', '而且', '另外',
            '首先', '其次', '最后', '总的来说', '总之', '综上所述'
        ]
        
        # 垃圾评论模式
        self.spam_patterns = [
            r'^好+$', r'^赞+$', r'^棒+$',
            r'^\w{1,3}$',  # 过短评论
            r'^[0-9]+$',   # 纯数字
            r'^[\u4e00-\u9fa5]{1,2}$'  # 1-2个汉字
        ]
    
    def calculate_length_score(self, text: str) -> float:
        """计算长度得分"""
        length = len(text.strip())
        if length < 10:
            return 0.2
        elif length < 20:
            return 0.5
        elif length < 50:
            return 0.8
        elif length < 200:
            return 1.0
        else:
            return 0.9  # 过长可能是复制粘贴
    
    def calculate_complexity_score(self, text: str) -> float:
        """计算复杂性得分"""
        # 词汇多样性
        words = list(jieba.cut(text))
        unique_words = len(set(words))
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
            
        diversity_score = unique_words / total_words
        
        # 句子结构复杂性
        complexity_count = sum(1 for indicator in self.complexity_indicators 
                             if indicator in text)
        complexity_score = min(complexity_count / 3, 1.0)
        
        # 标点符号多样性
        punctuation_count = len(re.findall(r'[，。！？；：]', text))
        punctuation_score = min(punctuation_count / 5, 1.0)
        
        return (diversity_score * 0.5 + complexity_score * 0.3 + 
                punctuation_score * 0.2)
    
    def calculate_information_score(self, text: str) -> float:
        """计算信息丰富度得分"""
        info_score = 0.0
        category_count = 0
        
        for category, keywords in self.info_keywords.items():
            if any(keyword in text for keyword in keywords):
                category_count += 1
        
        info_score = min(category_count / len(self.info_keywords), 1.0)
        
        # 数字信息（具体数据更有价值）
        number_count = len(re.findall(r'\d+', text))
        number_score = min(number_count / 3, 0.3)
        
        return info_score * 0.7 + number_score * 0.3
    
    def calculate_originality_score(self, text: str, 
                                  reference_texts: List[str] = None) -> float:
        """计算原创性得分"""
        # 检查垃圾评论模式
        for pattern in self.spam_patterns:
            if re.match(pattern, text.strip()):
                return 0.1
        
        # 与参考文本的相似性检查
        if reference_texts:
            embeddings = self.sentence_model.encode([text] + reference_texts)
            similarities = cosine_similarity([embeddings[0]], embeddings[1:])
            max_similarity = np.max(similarities)
            
            if max_similarity > 0.9:
                return 0.2
            elif max_similarity > 0.8:
                return 0.5
            elif max_similarity > 0.7:
                return 0.7
        
        return 1.0
    
    def calculate_sentiment_score(self, text: str) -> float:
        """计算情感合理性得分"""
        from snownlp import SnowNLP
        
        try:
            sentiment = SnowNLP(text).sentiments
            
            # 极端情感（过于积极或消极）可能不真实
            if sentiment > 0.9 or sentiment < 0.1:
                return 0.6
            elif sentiment > 0.8 or sentiment < 0.2:
                return 0.8
            else:
                return 1.0
        except:
            return 0.5
    
    def detect_copy_modification(self, text: str, 
                               reference_texts: List[str]) -> Dict:
        """检测复制修改行为"""
        results = {
            'is_copy_modified': False,
            'similarity_score': 0.0,
            'edit_distance_ratio': 0.0,
            'most_similar_text': None
        }
        
        if not reference_texts:
            return results
        
        # 计算语义相似度
        embeddings = self.sentence_model.encode([text] + reference_texts)
        similarities = cosine_similarity([embeddings[0]], embeddings[1:])
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[0][max_similarity_idx]
        
        # 计算编辑距离
        most_similar_text = reference_texts[max_similarity_idx]
        edit_distance = Levenshtein.distance(text, most_similar_text)
        max_len = max(len(text), len(most_similar_text))
        edit_ratio = 1 - (edit_distance / max_len) if max_len > 0 else 0
        
        results.update({
            'similarity_score': float(max_similarity),
            'edit_distance_ratio': edit_ratio,
            'most_similar_text': most_similar_text
        })
        
        # 判断是否为复制修改
        if max_similarity > 0.8 and edit_ratio > 0.7:
            results['is_copy_modified'] = True
        
        return results
    
    def evaluate_quality(self, text: str, 
                        reference_texts: List[str] = None) -> Dict:
        """综合评估评论质量"""
        # 计算各项得分
        length_score = self.calculate_length_score(text)
        complexity_score = self.calculate_complexity_score(text)
        sentiment_score = self.calculate_sentiment_score(text)
        information_score = self.calculate_information_score(text)
        originality_score = self.calculate_originality_score(text, reference_texts)
        
        # 计算综合得分
        total_score = (
            length_score * self.weights['length_score'] +
            complexity_score * self.weights['complexity_score'] +
            sentiment_score * self.weights['sentiment_score'] +
            information_score * self.weights['information_score'] +
            originality_score * self.weights['originality_score']
        )
        
        # 质量等级
        if total_score >= 0.8:
            quality_level = "优秀"
        elif total_score >= 0.6:
            quality_level = "良好"
        elif total_score >= 0.4:
            quality_level = "一般"
        else:
            quality_level = "较差"
        
        # 检测复制修改
        copy_detection = self.detect_copy_modification(text, reference_texts or [])
        
        return {
            'total_score': round(total_score, 3),
            'quality_level': quality_level,
            'detailed_scores': {
                'length_score': round(length_score, 3),
                'complexity_score': round(complexity_score, 3),
                'sentiment_score': round(sentiment_score, 3),
                'information_score': round(information_score, 3),
                'originality_score': round(originality_score, 3)
            },
            'copy_detection': copy_detection,
            'timestamp': datetime.now().isoformat()
        }
