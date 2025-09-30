import os
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

# --- 配置参数 ---
MODEL_NAME = "/root/autodl-tmp/lee/chinese-roberta-wwm-ext"
DATA_PATH = "/root/autodl-tmp/lee/广告非广告评论_4000条.csv"
OUTPUT_DIR = "/root/autodl-tmp/lee/results/ad_classifier_model"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots") # 确保这个目录存在
LOG_DIR = "/root/autodl-tmp/lee/logs"

NUM_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

# 确保输出和绘图目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- 设置中文字体，确保图表中文显示正常 ---
# 请将 chinese_font_path 替换为你AutoDL系统上实际找到的中文字体文件路径
# 我们之前确认的路径是 /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc
chinese_font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'

if os.path.exists(chinese_font_path):
    from matplotlib.font_manager import FontProperties
    font_prop = FontProperties(fname=chinese_font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    print(f"Using font: {font_prop.get_name()} from {chinese_font_path}")
else:
    print(f"Warning: Chinese font file not found at {chinese_font_path}. Chinese characters in plots might be missing.")
    # 回退到SimHei，但通常也不会成功，只是一个备用
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


# --- 数据加载与预处理 ---
print("Loading data...")
try:
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully.")

    # **新增：将 'id' 列重命名为 'label'，因为 'id' 实际上是你的标签列**
    df.rename(columns={'id': 'label'}, inplace=True)

    # 确保 'label' 列是数值类型 (现在它就是原来的 'id' 列)
    df['label'] = df['label'].astype(int)
    print(f"Original dataset size: {len(df)}")
    print(f"Original label distribution:\n{df['label'].value_counts()}")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()
except Exception as e:
    print(f"Error loading or processing data: {e}")
    # 打印具体的异常，这有助于调试
    import traceback
    traceback.print_exc()
    exit()

# 数据清洗（简单的去重和空值处理）
df.dropna(subset=['comment', 'label'], inplace=True)
df.drop_duplicates(subset=['comment'], inplace=True)
print(f"Dataset size after cleaning: {len(df)}")

# 划分训练集和评估集 (stratify 保证类别比例不变)
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

print("\n--- Data Split Summary (Before Oversampling) ---")
print(f"Train samples: {len(train_df)}")
print(f"  Ad (1) in train: {train_df['label'].sum()}")
print(f"  Non-Ad (0) in train: {len(train_df) - train_df['label'].sum()}")
print(f"Eval samples: {len(eval_df)}")
print(f"  Ad (1) in eval: {eval_df['label'].sum()}")
print(f"  Non-Ad (0) in eval: {len(eval_df) - eval_df['label'].sum()}")

# 针对训练集进行过采样
print("\nApplying RandomOverSampler to training data...")
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(train_df[['comment']], train_df['label'])
train_df_resampled = pd.DataFrame({'comment': X_train_resampled['comment'], 'label': y_train_resampled})

print("\n--- Data Split Summary (After Oversampling) ---")
print(f"Train samples (after oversampling): {len(train_df_resampled)}")
print(f"  Ad (1) in train (after oversampling): {train_df_resampled['label'].sum()}")
print(f"  Non-Ad (0) in train (after oversampling): {len(train_df_resampled) - train_df_resampled['label'].sum()}")
print(f"Eval samples: {len(eval_df)}")
print(f"  Ad (1) in eval: {eval_df['label'].sum()}")
print(f"  Non-Ad (0) in eval: {len(eval_df) - eval_df['label'].sum()}")


# 转换为 Hugging Face Dataset 格式
train_dataset = Dataset.from_pandas(train_df_resampled)
eval_dataset = Dataset.from_pandas(eval_df)

# --- 加载分词器和模型 ---
print("Loading tokenizer and model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 注意：这里我们第一次加载原始的预训练模型，需要 from_tf=True
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2, # 二分类任务：广告/非广告
        from_tf=True # 确保加载TF格式的权重
    )
    print("Tokenizer and model loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer or model from Hugging Face: {e}")
    exit()

# --- 分词函数 ---
def tokenize_function(examples):
    return tokenizer(examples["comment"], truncation=True, padding="max_length", max_length=512)

print("Tokenizing training data...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# 设置格式以匹配PyTorch训练

# 定义模型训练需要的列
columns_to_keep = ['input_ids', 'token_type_ids', 'attention_mask', 'label']

# 动态获取并移除不需要的列
# 对于训练集
columns_to_remove_train = [col for col in tokenized_train_dataset.column_names if col not in columns_to_keep]
tokenized_train_dataset = tokenized_train_dataset.remove_columns(columns_to_remove_train)

# 对于评估集
columns_to_remove_eval = [col for col in tokenized_eval_dataset.column_names if col not in columns_to_keep]
tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(columns_to_remove_eval)

tokenized_train_dataset.set_format("torch")
tokenized_eval_dataset.set_format("torch")

# --- 定义评估指标 ---
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', pos_label=1)
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- 训练参数和Trainer ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=LOG_DIR,
    logging_steps=100,
    evaluation_strategy="epoch", # 每个epoch结束时评估
    save_strategy="epoch", # 每个epoch结束时保存
    load_best_model_at_end=True, # 训练结束后加载最佳模型
    metric_for_best_model="f1", # 根据f1分数选择最佳模型
    greater_is_better=True,
    report_to="none", # 不上报到wandb等
    learning_rate=LEARNING_RATE,
    save_total_limit=1 # 只保留最新的一个检查点和最佳检查点
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# --- 开始训练 ---
print("\nStarting training...")
trainer.train()
print("Training finished.")

# 保存最终的模型（通常是加载的最佳模型）
# trainer.save_model(OUTPUT_DIR) # 这一行通常由 load_best_model_at_end 隐式处理，或由save_strategy保存检查点

# 评估最佳模型在评估集上的表现
print("Evaluating best model on evaluation set...")
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# --- 绘图功能 ---
def plot_metrics(log_history, plot_path):
    epochs = [entry['epoch'] for entry in log_history if 'eval_loss' in entry]
    eval_losses = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry]
    eval_accuracies = [entry['eval_accuracy'] for entry in log_history if 'eval_accuracy' in entry]
    eval_f1s = [entry['eval_f1'] for entry in log_history if 'eval_f1' in entry]

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, eval_losses, label='评估损失')
    plt.plot(epochs, eval_accuracies, label='评估准确率')
    plt.plot(epochs, eval_f1s, label='评估 F1-Score')
    plt.title('模型评估指标随 Epoch 变化')
    plt.xlabel('Epoch')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Training metrics plot saved to {plot_path}")

def plot_confusion_matrix(predictions, labels, plot_path, title='混淆矩阵'):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['非广告', '广告'], yticklabels=['非广告', '广告'])
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix plot saved to {plot_path}")

print("\nGenerating and saving plots...")
# 获取训练日志，包括评估指标
log_history = trainer.state.log_history
plot_metrics(log_history, os.path.join(PLOT_DIR, "training_metrics.png"))

# 获取评估集上的预测结果
eval_predictions = trainer.predict(tokenized_eval_dataset)
eval_preds = np.argmax(eval_predictions.predictions, axis=1)
eval_labels = eval_predictions.label_ids
plot_confusion_matrix(eval_preds, eval_labels, os.path.join(PLOT_DIR, "confusion_matrix.png"))


# --- Testing prediction with the trained model ---
print("\n--- Testing prediction with the trained model ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **核心修改：明确指定加载路径为 checkpoint-1500**
# 这是包含完整模型和分词器文件的目录
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "checkpoint-1500")

print(f"Using device: {device}")
print(f"Loading tokenizer from: {FINAL_MODEL_PATH}")
loaded_tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_PATH)

print(f"Loading model from: {FINAL_MODEL_PATH}")
loaded_model = AutoModelForSequenceClassification.from_pretrained(FINAL_MODEL_PATH)
loaded_model.to(device)
loaded_model.eval() # 设置为评估模式
print("Model loaded successfully for prediction.")

# 定义预测函数
def predict_single_comment(comment_text):
    inputs = loaded_tokenizer(comment_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = loaded_model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class_id = torch.argmax(probabilities, dim=1).item()

    label_map = {0: '非广告', 1: '广告'}
    predicted_label = label_map[predicted_class_id]

    return predicted_label, probabilities[0].tolist()

# 示例评论
test_reviews = [
    '这家酒店服务真的太棒了，强烈推荐！联系电话13812345678。',
    '景区很美，空气清新，值得一去。',
    '想体验真正的风情？找XX旅行社，全程无忧，门票打折！',
    '今天天气很好，玩得很开心。',
    '提供景区周边私人导游服务，价格优惠，欢迎咨询V信：ABCD123。',
    '这是一条非常普通的评论，没有任何广告内容，纯粹分享体验。',
    '我发现一个超棒的购物平台，点击这里链接：www.example.com',
    '这次旅行非常愉快，景色迷人，酒店也舒适，没有什么可抱怨的。',
    '我们公司最近搞活动，转发朋友圈送好礼，赶紧来参与！',
    '这个地方太棒了，下次还来！',
    '快来抢购我们的新品，限时优惠，错过再等一年！咨询热线：400-888-6666',
    '专业代写各类论文，价格合理，保质保量，详情加扣扣：123456789。',
    '恭喜您获得iPhone 16抽奖资格，点击链接领取：bit.ly/lucky-draw',
    '想学编程？报名我们的Python入门班，名师一对一指导，保你学会！',
    '全新房源，地铁口，精装修，拎包入住，看房请致电小王：13500001111。'
]

print("Predicting 15 reviews...")
for review in test_reviews:
    label, probs = predict_single_comment(review)
    print(f"Review: '{review}' -> Predicted: {label}")