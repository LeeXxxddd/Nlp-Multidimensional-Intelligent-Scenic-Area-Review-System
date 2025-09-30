import os
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties

# === 路径和字体设置 ===
input_dir = "G:/leo nlp/output"
output_dir = os.path.join(input_dir, "visualizations")
os.makedirs(output_dir, exist_ok=True)
font_path = "C:/Windows/Fonts/simhei.ttf"  # 替换为你电脑中的中文字体路径
my_font = FontProperties(fname=font_path)

# === 输入文件对应类别 ===
files = {
    "差评": "差评.txt",
    "好评": "好评.txt",
    "中性评论": "中性评论.txt",
    "广告": "广告.txt",
    "无效重复的评论": "无效重复的评论.txt",
    "有效评论": "有效评论.txt",
    "语义相似度高的评论": "语义相似度高的评论.txt",
}
color_schemes = ['Blues', 'Greens', 'Purples', 'Oranges', 'Reds', 'copper', 'bone']

# === 数据容器 ===
length_dict = {}
text_dict = {}
emotion_dict = {}
comment_counts = {}
mean_lengths = {}

# === 加载与预处理 ===
for i, (label, filename) in enumerate(files.items()):
    path = os.path.join(input_dir, filename)
    if not os.path.exists(path):
        print(f"跳过：未找到 {filename}")
        continue

    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # 保存数量、文本、长度
    comment_counts[label] = len(lines)
    text_dict[label] = lines
    lengths = [len(line) for line in lines]
    length_dict[label] = lengths
    mean_lengths[label] = np.mean(lengths)

    # === 生成词云 ===
    words = ' '.join(jieba.cut(' '.join(lines)))
    wc = WordCloud(
        font_path=font_path,
        background_color='white',
        colormap=color_schemes[i % len(color_schemes)],
        width=800,
        height=400
    ).generate(words)
    wc.to_file(os.path.join(output_dir, f"{label}_词云.png"))

    # === 情感分析（用 TextBlob 简单分析） ===
    pos, neg, neu = 0, 0, 0
    for line in lines:
        try:
            score = TextBlob(line).sentiment.polarity
            if score > 0.1:
                pos += 1
            elif score < -0.1:
                neg += 1
            else:
                neu += 1
        except:
            neu += 1
    emotion_dict[label] = [pos, neg, neu]

# === 图1：评论数量柱状图 ===
plt.figure(figsize=(10, 6))
plt.bar(comment_counts.keys(), comment_counts.values(), color='skyblue')
plt.title("评论数量柱状图", fontproperties=my_font)
plt.xticks(fontproperties=my_font)
plt.ylabel("数量", fontproperties=my_font)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "评论数量_柱状图.png"))
plt.close()

# === 图2：平均长度折线图 ===
plt.figure(figsize=(10, 6))
plt.plot(mean_lengths.keys(), mean_lengths.values(), marker='o', color='orange')
plt.title("评论平均长度折线图", fontproperties=my_font)
plt.xticks(rotation=45, fontproperties=my_font)
plt.ylabel("平均长度", fontproperties=my_font)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "平均长度_折线图.png"))
plt.close()

# === 图3：评论长度分布折线图（每类一条） ===
plt.figure(figsize=(12, 6))
for label, lengths in length_dict.items():
    plt.plot(range(len(lengths)), lengths, label=label)
plt.title("评论长度分布折线图", fontproperties=my_font)
plt.xlabel("评论序号", fontproperties=my_font)
plt.ylabel("长度", fontproperties=my_font)
plt.legend(prop=my_font)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "评论长度_折线图.png"))
plt.close()

# === 图4：评论长度箱线图 ===
plt.figure(figsize=(12, 6))
sns.boxplot(data=[length_dict[k] for k in files.keys()], orient='v')
plt.xticks(ticks=range(len(files)), labels=files.keys(), fontproperties=my_font, rotation=30)
plt.title("评论长度箱线图", fontproperties=my_font)
plt.ylabel("长度", fontproperties=my_font)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "评论长度_箱线图.png"))
plt.close()

# === 图5：情感极性柱状图 ===
labels = list(emotion_dict.keys())
pos_vals = [v[0] for v in emotion_dict.values()]
neg_vals = [v[1] for v in emotion_dict.values()]
neu_vals = [v[2] for v in emotion_dict.values()]

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, pos_vals, width=width, label='正面', color='lightgreen')
plt.bar(x, neu_vals, width=width, label='中性', color='gray')
plt.bar(x + width, neg_vals, width=width, label='负面', color='tomato')
plt.xticks(x, labels, fontproperties=my_font, rotation=30)
plt.ylabel("数量", fontproperties=my_font)
plt.title("情感极性柱状图", fontproperties=my_font)
plt.legend(prop=my_font)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "情感极性_柱状图.png"))
plt.close()

# === 图6：TF-IDF 高频词堆叠图 ===
all_texts = []
categories = []
for label, lines in text_dict.items():
    all_texts.extend(lines)
    categories.extend([label] * len(lines))

vectorizer = TfidfVectorizer(max_features=20, tokenizer=jieba.cut)
X = vectorizer.fit_transform(all_texts)
feature_names = vectorizer.get_feature_names_out()

df = pd.DataFrame(X.toarray(), columns=feature_names)
df['类别'] = categories
grouped = df.groupby("类别").mean()

grouped.T.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='tab20')
plt.title("TF-IDF 高频词堆叠图", fontproperties=my_font)
plt.ylabel("平均TF-IDF值", fontproperties=my_font)
plt.xticks(rotation=45, fontproperties=my_font)
plt.legend(prop=my_font)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "TFIDF_高频词堆叠图.png"))
plt.close()

# === 图7（可选）：语义相似度分布图（基于评论长度作为近似） ===
if "语义相似度高的评论" in length_dict:
    plt.figure(figsize=(10, 6))
    sns.histplot(length_dict["语义相似度高的评论"], bins=30, kde=True, color='purple')
    plt.title("语义相似度高的评论 - 长度分布（近似）", fontproperties=my_font)
    plt.xlabel("评论长度", fontproperties=my_font)
    plt.ylabel("频数", fontproperties=my_font)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "语义相似度_分布图.png"))
    plt.close()

print("✅ 所有图表已保存到:", output_dir)
