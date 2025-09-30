import re
import jieba
import jieba.posseg
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from gensim import corpora, models, similarities
from snownlp import SnowNLP
import os
from typing import List, Dict, Set, Tuple, Optional

class ReviewAnalyzer:
    """景区及酒店网评文本有效性分析工具"""

    def __init__(self,
                 stopwords_path: str = "data/stopwords_zh.txt",
                 ad_keywords_path: str = "data/网络关键词.txt",
                 user_dict_path: Optional[str] = None):
        # 加载停用词和广告关键词
        self.stopwords = self._load_words(stopwords_path)
        self.ad_keywords = self._load_words(ad_keywords_path)

        # 加载用户词典（可选）
        if user_dict_path:
            try:
                jieba.load_userdict(user_dict_path)
            except Exception as e:
                print(f"加载自定义词典失败: {e}")

        jieba.initialize()

    def _load_words(self, file_path: str) -> Set[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return set([line.strip() for line in f if line.strip()])
        except FileNotFoundError:
            print(f"警告: 文件 {file_path} 不存在，将使用空集合")
            return set()

    def remove_duplicates(self, reviews: List[str]) -> Tuple[List[str], List[str]]:
        unique_reviews, duplicate_reviews = [], []
        review_counts = Counter(reviews)
        for review, count in review_counts.items():
            cleaned = str(review).strip()
            unique_reviews.append(cleaned)
            if count > 1:
                duplicate_reviews.extend([cleaned] * (count - 1))
        return unique_reviews, duplicate_reviews

    def detect_similar_reviews(self, reviews: List[str], threshold: float = 0.8) -> List[int]:
        if not reviews:
            return []

        cut_list = []
        for review in tqdm(reviews, desc="分词处理"):
            cut_list.append([w for w in jieba.cut(str(review))])

        dictionary = corpora.Dictionary(cut_list)
        corpus = [dictionary.doc2bow(doc) for doc in cut_list]

        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]

        index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=len(dictionary))

        similar_indices = []
        for i in tqdm(range(len(corpus_tfidf)), desc="计算相似度"):
            sims = index[corpus_tfidf[i]]
            for j in range(i + 1, len(sims)):
                if sims[j] > threshold:
                    similar_indices.append(j)

        return sorted(set(similar_indices))

    def detect_ads(self, reviews: List[str], min_keywords: int = 3) -> Tuple[List[str], List[str]]:
        non_ad_reviews, ad_reviews = [], []
        for review in reviews:
            text = str(review)
            count = sum(1 for kw in self.ad_keywords if kw in text)
            if count >= min_keywords:
                ad_reviews.append(text)
            else:
                non_ad_reviews.append(text)
        return non_ad_reviews, ad_reviews

    def analyze_sentiment(self, reviews: List[str]) -> Tuple[List[str], List[str], List[str]]:
        positive_reviews, neutral_reviews, negative_reviews = [], [], []
        for review in tqdm(reviews, desc="情感分析"):
            try:
                text = str(review)
                sentiment = SnowNLP(text).sentiments
                if sentiment > 0.5:
                    positive_reviews.append(text)
                elif sentiment == 0.5:
                    neutral_reviews.append(text)
                else:
                    negative_reviews.append(text)
            except Exception as e:
                print(f"情感分析错误: {e}，评论: {review}")
                negative_reviews.append(str(review))
        return positive_reviews, neutral_reviews, negative_reviews

    def process_csv_file(self, file_path: str, column_name: str = "review") -> List[str]:
        try:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                print("[编码警告] UTF-8 解码失败，尝试使用 GBK 编码重新读取")
                df = pd.read_csv(file_path, encoding='gbk')

            if column_name not in df.columns:
                raise ValueError(f"列 '{column_name}' 不存在于文件中，可选列有：{list(df.columns)}")

            raw = df[column_name].dropna().tolist()
            reviews = [str(r).strip() for r in raw if isinstance(r, str) and str(r).strip()]

            print(f"[CSV读取] 读取有效评论 {len(reviews)} 条")
            return reviews

        except Exception as e:
            print(f"[CSV读取错误] {e}")
            return []

    def process_txt_file(self, file_path: str) -> List[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            reviews = [line.strip() for line in lines if isinstance(line, str) and line.strip()]
            print(f"[TXT读取] 读取有效评论 {len(reviews)} 条")
            return reviews
        except Exception as e:
            print(f"[TXT读取错误] {e}")
            return []

    def save_to_file(self, reviews: List[str], file_path: str) -> None:
        try:
            if not reviews:
                return
            with open(file_path, 'w', encoding='utf-8') as f:
                for review in reviews:
                    f.write(f"{review}\n")
            print(f"已保存 {len(reviews)} 条评论到 {file_path}")
        except Exception as e:
            print(f"保存文件时出错: {e}")
