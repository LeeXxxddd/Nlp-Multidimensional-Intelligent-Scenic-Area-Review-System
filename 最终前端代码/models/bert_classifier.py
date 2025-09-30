import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import json

class BERTReviewClassifier:
    """基于BERT的评论分类器"""
    
    def __init__(self, model_name="hfl/chinese-bert-wwm-ext"):
        """初始化分类器"""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 分类标签
        self.labels = {
            'quality': ['低质量', '中等质量', '高质量'],
            'type': ['垃圾评论', '广告', '正常评论'],
            'sentiment': ['负面', '中性', '正面']
        }
    
    def prepare_data(self, texts: List[str], labels: List[int], 
                    max_length: int = 512) -> Dict:
        """准备训练数据"""
        # 分词和编码
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def train_classifier(self, train_texts: List[str], train_labels: List[int],
                        val_texts: List[str] = None, val_labels: List[int] = None,
                        num_labels: int = 3, epochs: int = 3, 
                        learning_rate: float = 2e-5):
        """训练分类器"""
        # 初始化模型
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        # 准备数据
        train_data = self.prepare_data(train_texts, train_labels)
        
        # 优化器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            # 批量处理
            batch_size = 16
            for i in range(0, len(train_texts), batch_size):
                batch_input_ids = train_data['input_ids'][i:i+batch_size].to(self.device)
                batch_attention_mask = train_data['attention_mask'][i:i+batch_size].to(self.device)
                batch_labels = train_data['labels'][i:i+batch_size].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(train_texts) // batch_size)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        print("训练完成!")
    
    def predict(self, texts: List[str]) -> List[Dict]:
        """预测文本类别"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用train_classifier方法")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                # 编码
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # 预测
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
                
                predictions.append({
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': probabilities.cpu().numpy().tolist()[0]
                })
        
        return predictions
    
    def evaluate_model(self, test_texts: List[str], test_labels: List[int]) -> Dict:
        """评估模型性能"""
        predictions = self.predict(test_texts)
        predicted_labels = [pred['predicted_class'] for pred in predictions]
        
        # 计算准确率
        accuracy = sum(1 for true, pred in zip(test_labels, predicted_labels) 
                      if true == pred) / len(test_labels)
        
        # 分类报告
        report = classification_report(test_labels, predicted_labels, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(test_labels, predicted_labels).tolist()
        }