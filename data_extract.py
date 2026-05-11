import pandas as pd
import numpy as np
import os
import glob

# 将单文件路径改为文件夹路径
INPUT_DIR = "./csv_data"
OUTPUT_DIR = "./data"

def simple_clean(input_file, output_file):
    print(f"正在读取数据: {input_file} ...")
    df = pd.read_csv(input_file, dtype=str)
    
    tcp_cols = ['tcp.dstport', 'tcp.srcport', 'tcp.len']
    udp_cols = ['udp.srcport', 'udp.dstport', 'udp.length'] 
    special_cols = set(tcp_cols + udp_cols)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(';', ',', regex=False)

    # 减少打印频率，保持终端整洁
    # print("正在处理通用列...")
    for col in df.columns:
        if col not in special_cols:
            if df[col].dtype == object:
                df[col] = df[col].str.split(',').str[-1]

    # print("正在处理 TCP/UDP 互斥逻辑...")
    is_tcp = df['tcp.srcport'].notna() & (df['tcp.srcport'] != '')
    
    for col in udp_cols:
        df[col] = np.where(is_tcp, np.nan, df[col])

    for col in udp_cols:
        if df[col].dtype == object:
            split_val = df[col].str.split(',').str[-1]
            df[col] = np.where(is_tcp, np.nan, split_val)

    print(f"处理完成，正在保存到: {output_file}\n")
    df.to_csv(output_file, index=False)

def process_folder(input_dir, output_dir):
    # 如果输出文件夹不存在，则自动创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出文件夹: {output_dir}")

    # 获取输入文件夹下所有的 csv 文件
    search_pattern = os.path.join(input_dir, "*.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"在 {input_dir} 目录下没有找到任何 .csv 文件！")
        return

    print(f"共找到 {len(csv_files)} 个文件，开始批量处理...\n" + "-"*40)

    # 遍历并处理每一个文件
    for file_path in csv_files:
        # 提取当前文件的文件名 (例如: N3_Facebook_188_0910_094907.csv)
        file_name = os.path.basename(file_path)
        
        # 拼接对应的输出路径
        output_path = os.path.join(output_dir, file_name)
        
        # 调用核心清洗函数
        simple_clean(file_path, output_path)

if __name__ == "__main__":
    process_folder(INPUT_DIR, OUTPUT_DIR)
    print("-" * 40 + "\n所有文件批量处理完毕！")