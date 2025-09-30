import re
import jieba
import jieba.posseg
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from gensim import corpora, models, similarities  # 这些现在暂时不用，但为了兼容性先保留
from snownlp import SnowNLP
import os
from typing import List, Dict, Set, Tuple, Optional

# 新增导入 Sentence-BERT 库
from sentence_transformers import SentenceTransformer, util
import torch


class ReviewAnalyzer:
    """景区及酒店网评文本有效性分析工具"""

    def __init__(self,
                 stopwords_path: str = "stopwords_zh.txt",
                 ad_keywords_path: str = "网络关键词.txt",
                 user_dict_path: Optional[str] = None,
                 # 新增 Sentence-BERT 模型路径参数
                 sbert_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        初始化分析工具

        Args:
            stopwords_path: 停用词文件路径
            ad_keywords_path: 广告关键词文件路径
            user_dict_path: 自定义词典路径(可选)
            sbert_model_name: Sentence-BERT 模型名称或路径
        """
        # 加载停用词
        self.stopwords = self._load_words(stopwords_path)
        # 加载广告关键词
        self.ad_keywords = self._load_words(ad_keywords_path)
        # 加载自定义词典(可选)
        if user_dict_path:
            jieba.load_userdict(user_dict_path)
        # 配置jieba分词器 (Sentence-BERT内部处理分词，这里主要用于其他模块，如广告关键词匹配)
        jieba.initialize()

        # 加载 Sentence-BERT 模型
        print(f"正在加载 Sentence-BERT 模型: {sbert_model_name}...")
        try:
            self.sbert_model = SentenceTransformer(sbert_model_name)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.sbert_model.to(self.device)
            print(f"Sentence-BERT 模型加载成功，使用设备: {self.device}")
        except Exception as e:
            print(f"警告: Sentence-BERT 模型加载失败: {e}，相似度检测功能可能受影响。请检查模型名称或网络连接。")
            self.sbert_model = None  # 如果加载失败，将模型设为None

    def _load_words(self, file_path: str) -> Set[str]:
        """加载文本文件中的词语"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return set([line.strip() for line in f if line.strip()])
        except FileNotFoundError:
            print(f"警告: 文件 {file_path} 不存在，将使用空集合")
            return set()

    def remove_duplicates(self, reviews: List[str]) -> Tuple[List[str], List[str]]:
        """
        去除重复评论 (完全相同的字符串)

        Args:
            reviews: 评论列表

        Returns:
            去重后的评论列表和重复的评论列表
        """
        unique_reviews_set = set()
        unique_reviews_order = []  # 保持原始顺序
        duplicate_reviews = []

        for review in reviews:
            stripped_review = review.strip()
            if stripped_review not in unique_reviews_set:
                unique_reviews_set.add(stripped_review)
                unique_reviews_order.append(stripped_review)
            else:
                duplicate_reviews.append(review)  # 原始未去空白的评论

        return unique_reviews_order, duplicate_reviews

    def detect_similar_reviews(self, reviews: List[str], threshold: float = 0.8) -> List[int]:
        """
        检测语义相似度高的评论 (使用Sentence-BERT)

        Args:
            reviews: 评论列表
            threshold: 相似度阈值 (0.0到1.0之间)

        Returns:
            相似评论的索引列表 (基于传入 reviews 的索引)
        """
        if not reviews or self.sbert_model is None:
            print("警告: 评论列表为空或Sentence-BERT模型未加载，跳过相似度检测。")
            return []

        print(f"正在使用 Sentence-BERT 计算 {len(reviews)} 条评论的语义相似度...")

        # 1. 编码评论，获取 Embedding 向量
        # batch_size 可以根据你的GPU内存调整
        embeddings = self.sbert_model.encode(reviews,
                                             convert_to_tensor=True,
                                             show_progress_bar=True,
                                             batch_size=32)

        similar_indices = set()

        # 2. 计算两两相似度
        # 为了避免重复计算和自我比较，只计算上三角部分
        # util.pytorch_cos_sim 更高效
        for i in tqdm(range(len(embeddings)), desc="计算两两相似度"):
            # 只与它后面的评论进行比较
            # 这里使用了torch.tensor.to() 方法，确保数据在正确的设备上
            # 这里的util.pytorch_cos_sim 是更高效的批量计算方式，但它一次性计算所有pair的相似度
            # 如果reviews数量非常大，这个方法可能内存占用过高
            # 对于当前需求，我们继续使用循环，每次计算一个embedding与后续所有embedding的相似度，避免构建巨大的矩阵

            # 从当前 embedding 开始，向后计算相似度
            # 计算当前评论 i 与所有后续评论 j (j > i) 的余弦相似度
            cos_scores = util.pytorch_cos_sim(embeddings[i], embeddings[i + 1:])

            # 找到相似度超过阈值的评论索引
            # .squeeze(0) 是为了处理当只有一行数据时的维度问题
            # .cpu().numpy() 将结果转回Numpy数组以便处理
            # 加上 i+1 是因为 cos_scores 是从 i+1 索引开始计算的
            high_similarity_indices = torch.where(cos_scores > threshold)[1].cpu().numpy() + (i + 1)

            for idx in high_similarity_indices:
                similar_indices.add(idx)

        # 返回排序后的相似评论索引
        return sorted(list(similar_indices))

    def detect_ads(self, reviews: List[str], min_keywords: int = 3) -> Tuple[List[str], List[str]]:
        """
        检测广告评论 (目前仍基于关键词匹配，后续将升级为BERT分类器)

        Args:
            reviews: 评论列表
            min_keywords: 触发广告的最少关键词数量

        Returns:
            非广告评论列表和广告评论列表
        """
        non_ad_reviews = []
        ad_reviews = []

        for review in reviews:
            ad_count = 0
            for keyword in self.ad_keywords:
                if keyword in review:
                    ad_count += 1

            if ad_count >= min_keywords:
                ad_reviews.append(review)
            else:
                non_ad_reviews.append(review)

        return non_ad_reviews, ad_reviews

    def analyze_sentiment(self, reviews: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        情感分析，将评论分为好评、中性评论和差评 (使用SnowNLP)

        Args:
            reviews: 评论列表

        Returns:
            好评列表、中性评论列表、差评列表
        """
        positive_reviews = []
        neutral_reviews = []
        negative_reviews = []

        for review in tqdm(reviews, desc="情感分析"):
            try:
                sentiment = SnowNLP(review).sentiments
                if sentiment >= 0.65:  # 调整阈值，更严格的好评
                    positive_reviews.append(review)
                elif sentiment <= 0.35:  # 调整阈值，更严格的差评
                    negative_reviews.append(review)
                else:  # 0.35 < sentiment < 0.65 认为是中性
                    neutral_reviews.append(review)
            except Exception as e:
                print(f"情感分析错误: {e}，评论: {review}。将评论默认分类为中性。")
                neutral_reviews.append(review)  # 错误或无法分析时，默认分类为中性，而不是差评

        return positive_reviews, neutral_reviews, negative_reviews

    def process_csv_file(self, file_path: str, review_column: str = "review") -> List[str]:
        """
        从CSV文件中读取评论

        Args:
            file_path: CSV文件路径
            review_column: 评论所在列名

        Returns:
            评论列表
        """
        try:
            # 尝试多种编码读取
            encodings = ['utf-8', 'gb18030', 'gbk', 'latin-1']
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                raise ValueError(f"无法读取文件: {file_path}")

            if review_column not in df.columns:
                raise ValueError(f"文件 {file_path} 中未找到列: {review_column}")

            # 移除 NaN 值并转换为列表
            return df[review_column].dropna().astype(str).tolist()

        except Exception as e:
            print(f"处理CSV文件时出错: {e}")
            return []

    def process_txt_file(self, file_path: str) -> List[str]:
        """
        从TXT文件中读取评论

        Args:
            file_path: TXT文件路径

        Returns:
            评论列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 过滤空行并去除空白符
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"处理TXT文件时出错: {e}")
            return []

    def save_to_file(self, reviews: List[str], file_path: str) -> None:
        """
        将评论保存到文件

        Args:
            reviews: 评论列表
            file_path: 文件路径
        """
        if not reviews:
            print(f"没有评论可保存到 {file_path}")
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for review in reviews:
                    f.write(f"{review}\n")
            print(f"已保存 {len(reviews)} 条评论到 {file_path}")
        except Exception as e:
            print(f"保存文件时出错: {e}")

    def analyze_reviews(self, input_files: List[str], output_dir: str = "output",
                        review_column: str = "review", similarity_threshold: float = 0.8,
                        ad_keywords_threshold: int = 3) -> None:
        """
        分析评论文件并输出结果

        Args:
            input_files: 输入文件列表
            output_dir: 输出目录
            review_column: CSV文件中评论所在列名
            similarity_threshold: 相似度阈值
            ad_keywords_threshold: 广告关键词阈值
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 读取所有评论
        all_reviews = []
        for file in input_files:
            if file.lower().endswith('.csv'):
                reviews = self.process_csv_file(file, review_column)
                print(f"从 {file} 读取了 {len(reviews)} 条评论")
                all_reviews.extend(reviews)
            elif file.lower().endswith('.txt'):
                reviews = self.process_txt_file(file)
                print(f"从 {file} 读取了 {len(reviews)} 条评论")
                all_reviews.extend(reviews)
            else:
                print(f"忽略不支持的文件类型: {file}")

        print(f"总共处理 {len(all_reviews)} 条评论")

        # 步骤1: 去除完全重复的评论
        unique_reviews, duplicate_reviews = self.remove_duplicates(all_reviews)
        self.save_to_file(duplicate_reviews, f"{output_dir}/无效重复的评论.txt")
        print(f"去除了 {len(duplicate_reviews)} 条完全重复评论，剩余 {len(unique_reviews)} 条")

        # 步骤2: 检测语义相似度高的评论 (使用Sentence-BERT)
        # 注意：这里的similar_indices是unique_reviews中的索引
        similar_indices = self.detect_similar_reviews(unique_reviews, similarity_threshold)

        # 分离出相似评论和剩余评论
        similar_reviews_content = []
        remaining_reviews_after_similarity = []

        # 遍历unique_reviews，根据similar_indices判断是否为相似评论
        for i, review in enumerate(unique_reviews):
            if i in similar_indices:
                similar_reviews_content.append(review)
            else:
                remaining_reviews_after_similarity.append(review)

        self.save_to_file(similar_reviews_content, f"{output_dir}/语义相似度高的评论.txt")
        print(
            f"检测到 {len(similar_reviews_content)} 条语义相似度高的评论，剩余 {len(remaining_reviews_after_similarity)} 条")

        # 步骤3: 检测广告评论
        non_ad_reviews, ad_reviews = self.detect_ads(remaining_reviews_after_similarity, ad_keywords_threshold)
        self.save_to_file(ad_reviews, f"{output_dir}/广告.txt")
        print(f"检测到 {len(ad_reviews)} 条广告评论，剩余 {len(non_ad_reviews)} 条")

        # 步骤4: 情感分析
        positive, neutral, negative = self.analyze_sentiment(non_ad_reviews)
        self.save_to_file(positive, f"{output_dir}/好评.txt")
        self.save_to_file(neutral, f"{output_dir}/中性评论.txt")  # 将原来的“无用评论”更名为“中性评论”
        self.save_to_file(negative, f"{output_dir}/差评.txt")
        print(f"好评: {len(positive)}, 中性: {len(neutral)}, 差评: {len(negative)}")

        # 保存最终有效评论 (好评 + 差评)
        # 这里的“有效评论”定义为：非重复、非语义相似、非广告、非中性的评论（即带有明确情感的评论）
        effective_reviews = positive + negative
        self.save_to_file(effective_reviews, f"{output_dir}/有效评论.txt")
        print(f"总共筛选出 {len(effective_reviews)} 条最终有效评论 (好评+差评)")


def main():
    """主函数示例"""
    # 初始化分析器，这里可以指定一个中文的Sentence-BERT模型
    # 如果你觉得 'paraphrase-multilingual-MiniLM-L12-v2' 效果不够好，可以尝试 'moka-ai/m3e-base'
    analyzer = ReviewAnalyzer(
        stopwords_path="stopwords_zh.txt",
        ad_keywords_path="网络关键词.txt",
        sbert_model_name='paraphrase-multilingual-MiniLM-L12-v2'  # 或 'moka-ai/m3e-base'
    )

    # 待分析的文件列表
    input_files = [
        "景区评论.csv",  # 确保你的CSV文件在这个目录下
        # 可以添加更多文件
    ]

    # 执行分析
    analyzer.analyze_reviews(
        input_files=input_files,
        output_dir="output1",
        review_column="评论内容",  # CSV文件中评论所在列名
        similarity_threshold=0.8,  # Sentence-BERT 相似度阈值，通常0.7-0.9是一个好的范围
        ad_keywords_threshold=3  # 广告关键词阈值
    )


if __name__ == "__main__":
    main()