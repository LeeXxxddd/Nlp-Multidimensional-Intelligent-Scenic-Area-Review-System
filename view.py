import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from wordcloud import WordCloud
import jieba
import numpy as np
from collections import Counter

# --- 配置部分 ---
INPUT_DIR = r"G:\leo nlp\output"
OUTPUT_DIR = r"generated_visualizations"

FILE_DETAILS = {
    "positive": {"name": "好评.txt", "label": "好评"},
    "negative": {"name": "差评.txt", "label": "差评"},
    "neutral": {"name": "中性评论.txt", "label": "中性评论"},
    "advertisements": {"name": "广告.txt", "label": "广告"},
    "invalid_duplicate": {"name": "无效重复的评论.txt", "label": "无效重复的评论"},
    "valid": {"name": "有效评论.txt", "label": "有效评论"},
    "semantically_similar": {"name": "语义相似度高的评论.txt", "label": "语义相似度高的评论"}
}

CHINESE_FONT_MATPLOTLIB = 'SimHei'  # Matplotlib尝试使用的字体名称

# !!! 【至关重要】请通过本回复开头的独立测试脚本确定此路径的有效性 !!!
# !!! 并将其替换为您系统中一个【真实存在的、包含中文的、有效的 .ttf 或 .otf 字体文件】的【绝对路径】 !!!
# 示例: FONT_PATH_WORDCLOUD = r"C:\Windows\Fonts\simhei.ttf" # (黑体)
# 或: FONT_PATH_WORDCLOUD = r"C:\Windows\Fonts\msyh.ttf"   # (微软雅黑)
# 或: FONT_PATH_WORDCLOUD = r"C:\Windows\Fonts\Deng.ttf"    # (等线)
FONT_PATH_WORDCLOUD = r"C:\Windows\Fonts\simhei.ttf"  # <--- 请务必修改为您的有效路径!


# --- 辅助函数 (部分有优化或新增) ---
def count_lines_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return len([line for line in f if line.strip()])
    except FileNotFoundError:
        print(f"警告: 文件 {filepath} 未找到。计数为0。")
        return 0
    except Exception as e:
        print(f"读取文件 {filepath} 计数时出错: {e}")
        return 0


def read_text_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"警告: 文件 {filepath} 未找到。相关文本分析将跳过。")
        return ""
    except Exception as e:
        print(f"读取文件 {filepath} 时出错: {e}")
        return ""


def plot_bar_chart(categories, values, title, xlabel, ylabel, output_filename, is_horizontal=False, color=None):
    if not categories or not values or len(categories) != len(values):
        print(f"无法生成柱状图 '{title}': 数据不足或不匹配。")
        return

    plt.figure(figsize=(12, max(7, len(categories) * 0.55 if is_horizontal else 7)))
    if is_horizontal:
        bars = plt.barh(categories, values, color=color if color else plt.cm.get_cmap('Pastel2').colors[0])
        plt.xlabel(ylabel, fontsize=12)
        plt.ylabel(xlabel, fontsize=12)
        plt.gca().invert_yaxis()
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
    else:
        bars = plt.bar(categories, values, color=color if color else plt.cm.get_cmap('Pastel1').colors)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)

    plt.title(title, fontsize=16)
    plt.tight_layout()

    max_val = max(values) if values else 1
    for bar_idx, bar in enumerate(bars):
        if is_horizontal:
            val = bar.get_width()
            label_text = str(int(val))
            plt.text(val + 0.01 * max_val, bar.get_y() + bar.get_height() / 2,
                     label_text, va='center', ha='left', fontsize=9)
        else:
            val = bar.get_height()
            label_text = str(int(val))
            # 获取柱子颜色以决定标签颜色 (简单处理)
            bar_color = bar.get_facecolor()
            text_color = 'white' if sum(bar_color[:3]) < 1.5 else 'black'  # 简单亮度判断
            plt.text(bar.get_x() + bar.get_width() / 2.0, val + 0.01 * max_val,
                     label_text, ha='center', va='bottom', fontsize=9, color=text_color, fontweight='bold')

    plt.savefig(output_filename)
    plt.close()
    print(f"图表已保存为: {output_filename}")


def generate_wordcloud_func(text, title, output_filename, font_path_wc=FONT_PATH_WORDCLOUD):
    if not text.strip():
        print(f"无法为 '{title}' 生成词云: 文本内容为空。")
        return
    word_list = jieba.lcut(text)
    segmented_text = " ".join(word_list)
    if not segmented_text.strip():
        print(f"无法为 '{title}' 生成词云: 分词后内容为空。")
        return
    try:
        print(f"  WordCloud '{title}': 尝试使用字体 '{font_path_wc}'")  # 增加日志
        wordcloud_obj = WordCloud(width=800, height=400,
                                  background_color='white',
                                  font_path=font_path_wc,  # 使用传入的特定字体路径
                                  collocations=False,
                                  prefer_horizontal=0.95
                                  ).generate(segmented_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_obj, interpolation='bilinear')
        plt.axis("off")
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()
        print(f"词云 '{title}' 已保存为: {output_filename}")
    except FileNotFoundError:  # WordCloud本身不直接抛这个，但预先检查或包装器可能抛
        print(f"【词云错误】'{title}': 字体文件未在路径 '{font_path_wc}' 找到！请修复 FONT_PATH_WORDCLOUD。")
    except RuntimeError as e:
        if "cannot open resource" in str(e) or "not a TrueType font" in str(e):
            print(
                f"【词云错误】'{title}': WordCloud无法打开或使用字体资源 '{font_path_wc}'. 请用本回复开头的独立脚本测试字体路径。详细: {e}")
        else:
            print(f"【词云错误】'{title}': 生成时发生运行时错误。详细: {e}")
    except Exception as e:
        print(f"【词云错误】'{title}': 生成时发生未知错误。详细: {e}")


def get_review_lengths(filepath, word_segmenter=jieba.lcut):
    lengths = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    words = word_segmenter(stripped_line)
                    lengths.append(len(words))
    except FileNotFoundError:
        print(f"警告: 计算长度时文件 {filepath} 未找到。")
    except Exception as e:
        print(f"计算长度时读取文件 {filepath} 出错: {e}")
    return lengths


def plot_review_length_histogram(length_data_map, title, output_filename, bins=30):
    # length_data_map: {'标签1': [长度列表1], '标签2': [长度列表2]}
    if not length_data_map or not any(length_data_map.values()):
        print(f"没有足够的评论长度数据来生成直方图 '{title}'。")
        return

    plt.figure(figsize=(12, 7))

    all_lengths_flat = [l for lengths in length_data_map.values() if lengths for l in lengths]
    if not all_lengths_flat:
        print(f"所有类别的评论长度数据均为空，无法生成直方图 '{title}'。")
        return

    max_len = max(all_lengths_flat) if all_lengths_flat else 50
    bin_edges = np.linspace(0, min(max_len, 200), bins + 1)  # 限制最大长度以提高可读性

    num_categories = len(length_data_map)
    # 使用tab10色板，如果类别多于10个会循环
    colors = plt.cm.get_cmap('tab10', num_categories if num_categories <= 10 else 10).colors

    for i, (label, lengths) in enumerate(length_data_map.items()):
        if lengths:
            plt.hist(lengths, bins=bin_edges, alpha=0.7, label=label, color=colors[i % len(colors)])

    plt.title(title, fontsize=16)
    plt.xlabel("评论长度 (基于Jieba分词的词数)", fontsize=12)
    plt.ylabel("评论数量", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"直方图已保存为: {output_filename}")


def get_top_ngrams(text, n=2, top_k=15, word_segmenter=jieba.lcut):
    if not text.strip(): return []
    words = word_segmenter(text.strip())
    if len(words) < n: return []
    ngrams_list = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not ngrams_list: return []
    return Counter(ngrams_list).most_common(top_k)


def plot_100_stacked_bar_optimized(data_dict, title, output_filename):
    # data_dict 结构: {'好评': count, '中性': count, '差评': count} (顺序影响堆叠)
    labels_ordered = ['好评', '中性', '差评']  # 控制堆叠顺序
    counts_ordered = [data_dict.get(label, 0) for label in labels_ordered]

    total_counts = sum(counts_ordered)
    if total_counts == 0:
        print(f"无法生成100%堆叠柱状图 '{title}': 总计数为0。")
        return

    proportions = [c / total_counts * 100 for c in counts_ordered]

    fig, ax = plt.subplots(figsize=(6, 7))  # 调整了尺寸，单个柱子不需要太宽，但可能需要高一些以容纳图例

    # 使用Matplotlib的Set2或Pastel2色板，通常比较美观
    # colors = plt.cm.get_cmap('Pastel2', len(labels_ordered)).colors
    # 或者自定义颜色
    custom_colors = {'好评': '#77DD77', '中性': '#AEC6CF', '差评': '#FF6961'}  # 柔和的颜色
    bar_colors = [custom_colors.get(label, '#CCCCCC') for label in labels_ordered]

    bottom = 0
    for i, label in enumerate(labels_ordered):
        if counts_ordered[i] > 0:  # 只绘制有数据的部分
            ax.bar('情感比例', proportions[i], bottom=bottom, label=label,
                   color=bar_colors[i], edgecolor='white', linewidth=0.7)
            # 在柱子中间添加百分比文本
            if proportions[i] > 5:  # 只为占比大于5%的部分添加文本
                text_color = 'black'  # 对于这些柔和颜色，黑色文本通常OK
                ax.text('情感比例', bottom + proportions[i] / 2, f"{proportions[i]:.1f}%",
                        ha='center', va='center', color=text_color, fontsize=10, fontweight='bold')
            bottom += proportions[i]

    ax.set_ylabel("评论占比 (%)", fontsize=12)
    ax.set_xlabel("")  # X轴是类别标签，这里只有一个“情感比例”
    ax.set_xticks([])  # 移除X轴刻度
    ax.set_title(title, fontsize=16, pad=20)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=len(labels_ordered), frameon=False)
    plt.ylim(0, 100)
    plt.box(False)  # 移除图表边框
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # 调整布局为图例和标题留空间
    plt.savefig(output_filename)
    plt.close()
    print(f"优化的100%堆叠柱状图已保存为: {output_filename}")


# --- 主脚本 ---
if __name__ == "__main__":
    try:
        plt.rcParams['font.sans-serif'] = [CHINESE_FONT_MATPLOTLIB, 'Microsoft YaHei']  # 备选字体
        plt.rcParams['axes.unicode_minus'] = False
        print(f"Matplotlib尝试使用字体: {plt.rcParams['font.sans-serif']}")
    except Exception as e:
        print(f"警告: 设置matplotlib默认中文字体 '{CHINESE_FONT_MATPLOTLIB}' 时出错: {e}")

    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            print(f"输出目录已创建: {os.path.abspath(OUTPUT_DIR)}")
        except Exception as e:
            print(f"创建输出目录 '{OUTPUT_DIR}' 失败: {e}. 请检查权限或手动创建。")
            exit()

    print(f"\n--- 使用WordCloud字体路径: {FONT_PATH_WORDCLOUD} ---")
    if not os.path.exists(FONT_PATH_WORDCLOUD):
        print(f"【严重警告】WordCloud字体文件路径 '{FONT_PATH_WORDCLOUD}' 无效或文件不存在! 词云图很可能会失败。")
        print("请务必修改脚本中的 FONT_PATH_WORDCLOUD 为一个正确的字体文件绝对路径。")
        print("您可以先运行本回复开头的【独立字体测试脚本】来确认您的字体路径。")

    category_counts = {}
    print("\n--- 正在读取文件计数 ---")
    for key, details in FILE_DETAILS.items():
        filepath = os.path.join(INPUT_DIR, details["name"])
        category_counts[details["label"]] = count_lines_in_file(filepath)

    print("\n--- 各类别评论数量 ---")
    has_any_data = any(c > 0 for c in category_counts.values())
    for label, count in category_counts.items(): print(f"{label}: {count}")
    if not has_any_data:
        print("\n所有类别文件均为空或未找到，无法生成图表。脚本终止。")
        exit()

    # --- 原有基础图表 ---
    print("\n--- 正在生成基础图表 ---")
    # 01 整体构成
    # ... (代码与上一版类似，这里略去以减少篇幅，但假设它存在且能运行)
    overall_labels = [label for label, count in category_counts.items() if count > 0]
    overall_values = [count for count in category_counts.values() if count > 0]
    if overall_labels:
        sorted_indices = sorted(range(len(overall_values)), key=lambda k: overall_values[k], reverse=True)
        plot_bar_chart(
            categories=[overall_labels[i] for i in sorted_indices],
            values=[overall_values[i] for i in sorted_indices],
            title="景区评论整体构成分析", xlabel="评论类别", ylabel="评论数量",
            output_filename=os.path.join(OUTPUT_DIR, "01_overall_composition_bar_chart.png")
        )
    # 02 情感分布 (计数)
    sentiment_labels_for_bar = [FILE_DETAILS["positive"]["label"], FILE_DETAILS["negative"]["label"],
                                FILE_DETAILS["neutral"]["label"]]
    sentiment_values_for_bar = [category_counts.get(label, 0) for label in sentiment_labels_for_bar]
    filtered_sentiment_labels = [label for i, label in enumerate(sentiment_labels_for_bar) if
                                 sentiment_values_for_bar[i] > 0]
    filtered_sentiment_values = [value for value in sentiment_values_for_bar if value > 0]
    if filtered_sentiment_labels:
        plot_bar_chart(
            categories=filtered_sentiment_labels, values=filtered_sentiment_values,
            title="评论情感分布（好评、差评、中性）", xlabel="情感类别", ylabel="评论数量",
            output_filename=os.path.join(OUTPUT_DIR, "02_sentiment_distribution_bar_chart.png")
        )
    # 03 无效广告计数
    other_cat_labels = [FILE_DETAILS["advertisements"]["label"], FILE_DETAILS["invalid_duplicate"]["label"]]
    other_cat_values = [category_counts.get(label, 0) for label in other_cat_labels]
    filtered_other_labels = [label for i, label in enumerate(other_cat_labels) if other_cat_values[i] > 0]
    filtered_other_values = [value for value in other_cat_values if value > 0]
    if filtered_other_labels:
        plot_bar_chart(
            categories=filtered_other_labels, values=filtered_other_values,
            title="广告与无效重复评论数量", xlabel="评论类别", ylabel="评论数量",
            output_filename=os.path.join(OUTPUT_DIR, "03_invalid_ads_counts_bar_chart.png")
        )

    # 04 好评词云 (使用 FONT_PATH_WORDCLOUD)
    print("\n--- 尝试生成好评词云 ---")
    positive_text = read_text_from_file(os.path.join(INPUT_DIR, FILE_DETAILS["positive"]["name"]))
    if positive_text:
        generate_wordcloud_func(positive_text, "好评高频词云",
                                os.path.join(OUTPUT_DIR, "04_positive_reviews_wordcloud.png"),
                                font_path_wc=FONT_PATH_WORDCLOUD)
    else:
        print("好评文件为空或未找到，跳过词云。")

    # 05 差评词云 (使用 FONT_PATH_WORDCLOUD)
    print("\n--- 尝试生成差评词云 ---")
    negative_text = read_text_from_file(os.path.join(INPUT_DIR, FILE_DETAILS["negative"]["name"]))
    if negative_text:
        generate_wordcloud_func(negative_text, "差评高频词云",
                                os.path.join(OUTPUT_DIR, "05_negative_reviews_wordcloud.png"),
                                font_path_wc=FONT_PATH_WORDCLOUD)
    else:
        print("差评文件为空或未找到，跳过词云。")

    # --- 第一批扩展图表 ---
    print("\n--- 正在生成第一批扩展图表 ---")
    # 06 评论长度分布 (扩展了类别)
    print("正在分析评论长度...")
    length_data_map = {}
    categories_for_length = {
        "positive": FILE_DETAILS["positive"],
        "negative": FILE_DETAILS["negative"],
        "neutral": FILE_DETAILS["neutral"],
        "advertisements": FILE_DETAILS["advertisements"],  # 新增
        "invalid_duplicate": FILE_DETAILS["invalid_duplicate"]  # 新增
    }
    for key, details in categories_for_length.items():
        lengths = get_review_lengths(os.path.join(INPUT_DIR, details["name"]))
        if lengths: length_data_map[details["label"]] = lengths

    if length_data_map:
        plot_review_length_histogram(
            length_data_map,
            title="各类评论长度分布对比",  # 更新标题
            output_filename=os.path.join(OUTPUT_DIR, "06_review_length_histogram_extended.png")
        )
    else:
        print("用于长度分析的评论文件均为空或无法读取，跳过评论长度直方图。")

    # 07 好评高频二元组
    print("\n正在分析好评高频二元组...")
    if positive_text:
        top_pos_bigrams = get_top_ngrams(positive_text, n=2, top_k=15)
        if top_pos_bigrams:
            plot_bar_chart(categories=[bg[0] for bg in top_pos_bigrams], values=[bg[1] for bg in top_pos_bigrams],
                           title="好评高频二元组 (Top 15)", xlabel="二元组", ylabel="频率",
                           output_filename=os.path.join(OUTPUT_DIR, "07_positive_top_bigrams.png"), is_horizontal=True)
        else:
            print("未能从好评中提取有效二元组。")
    # 08 差评高频二元组
    print("\n正在分析差评高频二元组...")
    if negative_text:
        top_neg_bigrams = get_top_ngrams(negative_text, n=2, top_k=15)
        if top_neg_bigrams:
            plot_bar_chart(categories=[bg[0] for bg in top_neg_bigrams], values=[bg[1] for bg in top_neg_bigrams],
                           title="差评高频二元组 (Top 15)", xlabel="二元组", ylabel="频率",
                           output_filename=os.path.join(OUTPUT_DIR, "08_negative_top_bigrams.png"), is_horizontal=True)
        else:
            print("未能从差评中提取有效二元组。")

    # 09 优化的100%堆叠柱状图 - 情感比例
    print("\n正在生成优化的100%情感比例堆叠柱状图...")
    sentiment_counts_for_stacked = {
        FILE_DETAILS["positive"]["label"]: category_counts.get(FILE_DETAILS["positive"]["label"], 0),
        FILE_DETAILS["neutral"]["label"]: category_counts.get(FILE_DETAILS["neutral"]["label"], 0),
        FILE_DETAILS["negative"]["label"]: category_counts.get(FILE_DETAILS["negative"]["label"], 0)
    }
    if sum(sentiment_counts_for_stacked.values()) > 0:
        plot_100_stacked_bar_optimized(
            sentiment_counts_for_stacked,  # 函数内部会按预设顺序处理
            title="评论情感构成比例 (优化版)",
            output_filename=os.path.join(OUTPUT_DIR, "09_sentiment_stacked_100_percent_optimized.png")
        )
    else:
        print("情感数据不足以生成100%堆叠柱状图。")

    # --- 第二批扩展图表 (新增) ---
    print("\n--- 正在生成第二批扩展图表 ---")
    # 10 广告内容词云 (使用 FONT_PATH_WORDCLOUD)
    print("\n--- 尝试生成广告内容词云 ---")
    ads_text = read_text_from_file(os.path.join(INPUT_DIR, FILE_DETAILS["advertisements"]["name"]))
    if ads_text:
        generate_wordcloud_func(ads_text, "广告内容高频词云",
                                os.path.join(OUTPUT_DIR, "10_advertisement_wordcloud.png"),
                                font_path_wc=FONT_PATH_WORDCLOUD)
    else:
        print("广告文件为空或未找到，跳过广告词云。")

    # 11 好评高频三元组
    print("\n正在分析好评高频三元组...")
    if positive_text:
        top_pos_trigrams = get_top_ngrams(positive_text, n=3, top_k=15)
        if top_pos_trigrams:
            plot_bar_chart(categories=[tg[0] for tg in top_pos_trigrams], values=[tg[1] for tg in top_pos_trigrams],
                           title="好评高频三元组 (Top 15)", xlabel="三元组", ylabel="频率",
                           output_filename=os.path.join(OUTPUT_DIR, "11_positive_top_trigrams.png"), is_horizontal=True)
        else:
            print("未能从好评中提取有效三元组。")

    # 12 差评高频三元组
    print("\n正在分析差评高频三元组...")
    if negative_text:
        top_neg_trigrams = get_top_ngrams(negative_text, n=3, top_k=15)
        if top_neg_trigrams:
            plot_bar_chart(categories=[tg[0] for tg in top_neg_trigrams], values=[tg[1] for tg in top_neg_trigrams],
                           title="差评高频三元组 (Top 15)", xlabel="三元组", ylabel="频率",
                           output_filename=os.path.join(OUTPUT_DIR, "12_negative_top_trigrams.png"), is_horizontal=True)
        else:
            print("未能从差评中提取有效三元组。")

    print("\n--- 所有可视化代码执行完毕 ---")
    # ... (结束语和提示，与上一版类似)
    if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
        print(f"图表已尝试保存到 '{os.path.abspath(OUTPUT_DIR)}' 目录中。")
    elif has_any_data:
        print(f"执行完毕，但似乎没有图表成功保存。请仔细检查上述控制台输出的【错误】或【警告】信息，特别是关于字体路径的部分。")

    print("\n【重要最终提示】:")
    print(
        "1. 【词云字体】: 如果词云图仍失败并提示'cannot open resource'或类似字体错误，请务必使用本回复开头的【独立字体测试脚本】来严格确认您的 FONT_PATH_WORDCLOUD 变量所指向的字体文件路径是100%正确的、可访问的、且包含中文的。这是解决词云问题的关键。")
    print("2. 【依赖库】: 确保 matplotlib, wordcloud, jieba, numpy 已正确安装 ('pip install ...')。")
    print("3. 【输入文件】: 确认 INPUT_DIR (G:\\leo nlp\\output\\visualizations\\) 及内部的 .txt 文件路径和内容。")
    print(
        "4. 【更多高级图表】: 如时间序列分析（需时间戳）、主题模型可视化（需先进行主题提取）、网络图等，需要更复杂的数据预处理和特定库，可在明确数据和需求后进一步探讨。")