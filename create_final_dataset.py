import pandas as pd
import os
import random
import re

def read_text_file(file_path, encoding_list=['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin1']):
    """尝试使用不同编码读取文本文件"""
    for encoding in encoding_list:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.readlines()
            print(f"成功使用 {encoding} 编码读取文件: {file_path}")
            return [line.strip() for line in content if line.strip()]
        except Exception as e:
            print(f"尝试使用 {encoding} 编码读取文件失败: {e}")
    
    print(f"错误: 无法读取文件 {file_path}")
    return []

def read_csv_file(file_path, encoding_list=['utf-8-sig', 'gbk', 'gb2312', 'utf-8', 'latin1']):
    """尝试使用不同编码读取CSV文件"""
    for encoding in encoding_list:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取文件: {file_path}")
            return df
        except Exception as e:
            print(f"尝试使用 {encoding} 编码读取文件失败: {e}")
    
    print(f"错误: 无法读取文件 {file_path}")
    return pd.DataFrame()

def clean_comment(comment):
    """清理评论文本，去除序号和多余的引号"""
    # 去除开头的数字和点
    comment = re.sub(r'^(\d+\.?\s*)', '', comment)
    # 去除开头和结尾的引号
    comment = re.sub(r'^[""\']+|[""\']+$', '', comment)
    return comment.strip()

def create_final_dataset():
    base_dir = "d:\\leo nlp"
    output_dir = os.path.join(base_dir, "output")
    
    # 设置输入和输出文件路径
    positive_path = os.path.join(output_dir, "好评.txt")
    neutral_path = os.path.join(output_dir, "中性评论.txt")
    negative_path = os.path.join(output_dir, "差评.txt")
    
    # 广告评论文件在主目录下，不在output目录中
    ad_path = os.path.join(base_dir, "广告评论_1000条new_fixed.csv")
    
    # 尝试多个可能的广告评论文件名
    if not os.path.exists(ad_path):
        alternative_paths = [
            os.path.join(base_dir, "广告评论_1000条.csv"),
            os.path.join(base_dir, "广告评论_新数据_fixed.csv"),
            os.path.join(base_dir, "广告评论_合并_fixed.csv"),
            os.path.join(base_dir, "广告评论.csv")
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                ad_path = alt_path
                print(f"使用替代广告评论文件: {ad_path}")
                break
        else:
            print("警告: 未找到广告评论文件，请确保文件存在")
    
    # 输出文件路径
    non_ad_output = os.path.join(base_dir, "非广告评论_3000条.csv")
    combined_output = os.path.join(base_dir, "广告非广告评论_4000条.csv")
    
    print("开始处理数据...")
    
    # 读取非广告评论
    print("读取非广告评论文件...")
    positive_comments = read_text_file(positive_path)
    neutral_comments = read_text_file(neutral_path)
    negative_comments = read_text_file(negative_path)
    
    print(f"读取到好评: {len(positive_comments)}条")
    print(f"读取到中性评论: {len(neutral_comments)}条")
    print(f"读取到差评: {len(negative_comments)}条")
    
    # 合并所有非广告评论
    all_non_ad_comments = []
    all_non_ad_comments.extend([(clean_comment(comment), 0) for comment in positive_comments])
    all_non_ad_comments.extend([(clean_comment(comment), 0) for comment in neutral_comments])
    all_non_ad_comments.extend([(clean_comment(comment), 0) for comment in negative_comments])
    
    print(f"合并后的非广告评论总数: {len(all_non_ad_comments)}条")
    
    # 随机打乱并抽取3000条非广告评论
    random.shuffle(all_non_ad_comments)
    non_ad_sample = all_non_ad_comments[:3000] if len(all_non_ad_comments) > 3000 else all_non_ad_comments
    print(f"随机抽取的非广告评论数: {len(non_ad_sample)}条")
    
    # 创建非广告评论DataFrame
    non_ad_df = pd.DataFrame(non_ad_sample, columns=['comment', 'id'])
    
    # 读取广告评论
    print(f"读取广告评论文件: {ad_path}")
    ad_df = read_csv_file(ad_path)
    
    if len(ad_df) == 0:
        print("错误: 无法读取广告评论文件")
        return
    
    print(f"读取到广告评论: {len(ad_df)}条")
    
    # 确保广告评论DataFrame的列名正确
    if 'comment' not in ad_df.columns and len(ad_df.columns) >= 2:
        # 假设第二列是评论内容
        ad_df.rename(columns={ad_df.columns[1]: 'comment'}, inplace=True)
    
    if 'id' not in ad_df.columns and len(ad_df.columns) >= 1:
        # 假设第一列是id
        ad_df.rename(columns={ad_df.columns[0]: 'id'}, inplace=True)
    
    # 确保广告评论的id全部为1
    ad_df['id'] = 1
    
    # 只保留需要的列
    if 'comment' in ad_df.columns and 'id' in ad_df.columns:
        ad_df = ad_df[['id', 'comment']]
    else:
        print("错误: 广告评论文件缺少必要的列")
        return
    
    # 保存非广告评论CSV
    try:
        non_ad_df = non_ad_df[['id', 'comment']]  # 确保列顺序一致
        non_ad_df.to_csv(non_ad_output, index=False, encoding='utf-8-sig')
        print(f"成功保存非广告评论到: {non_ad_output}")
    except Exception as e:
        print(f"保存非广告评论文件时出错: {e}")
    
    # 合并广告和非广告评论
    combined_df = pd.concat([ad_df, non_ad_df], ignore_index=True)
    
    # 随机打乱合并后的数据
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 保存合并后的CSV
    try:
        combined_df.to_csv(combined_output, index=False, encoding='utf-8-sig')
        print(f"成功保存合并数据到: {combined_output}")
        print(f"合并数据总条数: {len(combined_df)}条")
        print(f"其中广告评论: {len(ad_df)}条, 非广告评论: {len(non_ad_df)}条")
    except Exception as e:
        print(f"保存合并文件时出错: {e}")
    
    # 显示数据示例
    print("\n非广告评论示例:")
    print(non_ad_df.head(3))
    
    print("\n合并数据示例:")
    print(combined_df.head(3))

if __name__ == "__main__":
    create_final_dataset()