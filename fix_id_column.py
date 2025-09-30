import pandas as pd
import os

def fix_id_column(file_path):
    """
    将CSV文件中的id列全部修改为1
    
    参数:
        file_path: CSV文件路径
    """
    print(f"正在处理文件: {file_path}")
    
    # 尝试不同的编码读取文件
    encodings = ['utf-8-sig', 'gbk', 'gb2312', 'utf-8', 'latin1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取文件")
            break
        except Exception as e:
            print(f"尝试使用 {encoding} 编码读取文件失败: {e}")
    
    if df is None:
        print("错误: 无法读取文件，请检查文件路径和编码")
        return
    
    print(f"原始数据形状: {df.shape}")
    print(f"原始列名: {df.columns.tolist()}")
    
    # 检查是否存在id列
    if 'id' in df.columns:
        # 将id列全部修改为1
        df['id'] = 1
        print("已将id列全部修改为1")
    else:
        print("警告: 文件中不存在id列，尝试查找其他可能的标签列")
        # 尝试查找其他可能的标签列
        possible_label_columns = ['label', 'class', 'category', 'type']
        found = False
        
        for col in possible_label_columns:
            if col in df.columns:
                print(f"找到可能的标签列: {col}，将其重命名为id并设置为1")
                df.rename(columns={col: 'id'}, inplace=True)
                df['id'] = 1
                found = True
                break
        
        if not found:
            print("未找到标签列，创建新的id列并设置为1")
            df['id'] = 1
    
    # 确保id列在第一位
    if 'id' in df.columns and list(df.columns).index('id') != 0:
        cols = df.columns.tolist()
        cols.remove('id')
        cols.insert(0, 'id')
        df = df[cols]
        print("已将id列移动到第一位")
    
    # 保存修改后的文件
    output_path = os.path.splitext(file_path)[0] + "_fixed.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"已保存修改后的文件: {output_path}")
    print(f"修改后数据形状: {df.shape}")
    print(f"修改后列名: {df.columns.tolist()}")
    print("\n数据示例:")
    print(df.head(3))

if __name__ == "__main__":
    # 默认处理生成的1000条广告评论文件
    file_path = "d:\\leo nlp\\广告评论_1000条new.csv"
    
    # 如果需要处理其他文件，可以在这里修改文件路径
    # file_path = "d:\\leo nlp\\其他文件.csv"
    
    fix_id_column(file_path)
