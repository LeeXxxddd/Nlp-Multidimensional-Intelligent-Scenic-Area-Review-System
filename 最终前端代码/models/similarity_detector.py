import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import Levenshtein
from typing import List, Dict, Tuple, Set
import networkx as nx

class SimilarityDetector:
    """高级相似度检测器"""
    
    def __init__(self, model_name="uer/sbert-base-chinese-nli"):
        """初始化检测器"""
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = 0.8
        self.edit_distance_threshold = 0.7
    
    def calculate_semantic_similarity(self, texts: List[str]) -> np.ndarray:
        """计算语义相似度矩阵"""
        embeddings = self.model.encode(texts)
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    
    def calculate_edit_distance_matrix(self, texts: List[str]) -> np.ndarray:
        """计算编辑距离相似度矩阵"""
        n = len(texts)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    distance_matrix[i][j] = 1.0
                else:
                    # 计算编辑距离相似度
                    distance = Levenshtein.distance(texts[i], texts[j])
                    max_len = max(len(texts[i]), len(texts[j]))
                    similarity = 1 - (distance / max_len) if max_len > 0 else 0
                    distance_matrix[i][j] = similarity
                    distance_matrix[j][i] = similarity
        
        return distance_matrix
    
    def detect_similar_clusters(self, texts: List[str], 
                              method: str = 'combined') -> List[List[int]]:
        """检测相似文本聚类"""
        if method == 'semantic':
            similarity_matrix = self.calculate_semantic_similarity(texts)
        elif method == 'edit_distance':
            similarity_matrix = self.calculate_edit_distance_matrix(texts)
        else:  # combined
            semantic_sim = self.calculate_semantic_similarity(texts)
            edit_sim = self.calculate_edit_distance_matrix(texts)
            similarity_matrix = (semantic_sim + edit_sim) / 2
        
        # 转换为距离矩阵用于聚类
        distance_matrix = 1 - similarity_matrix
        
        # 可能有小数精度误差导致负值（如 -1e-8），需裁剪
        distance_matrix[distance_matrix < 0] = 0.0
        # 使用DBSCAN聚类
        clustering = DBSCAN(
            eps=1-self.similarity_threshold, 
            min_samples=2, 
            metric='precomputed'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # 组织聚类结果
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label != -1:  # -1表示噪声点
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)
        
        return list(clusters.values())
    
    def detect_copy_modification_network(self, texts: List[str]) -> Dict:
        """基于网络分析检测复制修改"""
        # 构建相似度图
        semantic_sim = self.calculate_semantic_similarity(texts)
        edit_sim = self.calculate_edit_distance_matrix(texts)
        
        # 创建图
        G = nx.Graph()
        G.add_nodes_from(range(len(texts)))
        
        # 添加边
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                # 综合相似度
                combined_sim = (semantic_sim[i][j] + edit_sim[i][j]) / 2
                
                # 检测可能的复制修改
                if (semantic_sim[i][j] > 0.8 and edit_sim[i][j] > 0.7 and
                    semantic_sim[i][j] - edit_sim[i][j] > 0.1):
                    G.add_edge(i, j, weight=combined_sim, type='copy_modified')
                elif combined_sim > self.similarity_threshold:
                    G.add_edge(i, j, weight=combined_sim, type='similar')
        
        # 检测连通组件
        components = list(nx.connected_components(G))
        
        # 分析每个组件
        results = {
            'copy_modified_groups': [],
            'similar_groups': [],
            'suspicious_patterns': []
        }
        
        for component in components:
            if len(component) > 1:
                component_list = list(component)
                edges = G.edges(component, data=True)
                
                # 检查复制修改模式
                copy_modified_edges = [e for e in edges if e[2]['type'] == 'copy_modified']
                
                if copy_modified_edges:
                    results['copy_modified_groups'].append({
                        'indices': component_list,
                        'texts': [texts[i] for i in component_list],
                        'relationships': [(e[0], e[1], e[2]['weight']) for e in copy_modified_edges]
                    })
                else:
                    results['similar_groups'].append({
                        'indices': component_list,
                        'texts': [texts[i] for i in component_list]
                    })
        
        return results
    
    def analyze_modification_patterns(self, original: str, modified: str) -> Dict:
        """分析修改模式"""
        # 字符级别差异
        char_diff = Levenshtein.editops(original, modified)
        
        # 词级别差异
        import jieba
        original_words = list(jieba.cut(original))
        modified_words = list(jieba.cut(modified))
        word_diff = Levenshtein.editops(original_words, modified_words)
        
        # 分析修改类型
        modification_types = {
            'char_insertions': 0,
            'char_deletions': 0,
            'char_substitutions': 0,
            'word_insertions': 0,
            'word_deletions': 0,
            'word_substitutions': 0
        }
        
        for op in char_diff:
            if op[0] == 'insert':
                modification_types['char_insertions'] += 1
            elif op[0] == 'delete':
                modification_types['char_deletions'] += 1
            elif op[0] == 'replace':
                modification_types['char_substitutions'] += 1
        
        for op in word_diff:
            if op[0] == 'insert':
                modification_types['word_insertions'] += 1
            elif op[0] == 'delete':
                modification_types['word_deletions'] += 1
            elif op[0] == 'replace':
                modification_types['word_substitutions'] += 1
        
        # 计算修改程度
        total_char_ops = sum([modification_types['char_insertions'],
                             modification_types['char_deletions'],
                             modification_types['char_substitutions']])
        
        modification_ratio = total_char_ops / max(len(original), len(modified))
        
        return {
            'modification_types': modification_types,
            'modification_ratio': modification_ratio,
            'is_minor_modification': modification_ratio < 0.3,
            'char_differences': len(char_diff),
            'word_differences': len(word_diff)
        }
