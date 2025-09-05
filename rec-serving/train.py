import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from lightfm import LightFM
import os

def load_data():
    """加载 MovieLens 100k 数据集"""
    print("Step 1: Loading MovieLens 100k data...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'ml-100k')
    data_file = os.path.join(data_dir, 'u.data')
    
    try:
        df = pd.read_csv(data_file, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file}")
        print("Please ensure you have downloaded the ml-100k dataset and placed it correctly inside the 'rec-serving' directory.")
        exit()

    print("Data loaded successfully.")
    print("Data sample (first 5 rows):")
    print(df.head())
    
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()
    print(f"Number of unique users: {num_users}, Number of unique items: {num_items}")
    
    return df, num_users, num_items

def create_interaction_matrix(df, num_users, num_items):
    """根据 DataFrame 创建用户-物品交互稀疏矩阵"""
    print("\nStep 2: Creating user-item interaction matrix...")
    
    # LightFM 模型要求 user_id 和 item_id 是从 0 开始的。
    # MovieLens 数据集是从 1 开始的，所以我们需要减 1。
    df['user_id'] = df['user_id'] - 1
    df['item_id'] = df['item_id'] - 1

    # 创建稀疏矩阵
    # coo_matrix 接收三个参数: (数据, (行索引, 列索引))
    interactions = coo_matrix((df['rating'], (df['user_id'], df['item_id'])), 
                              shape=(num_users, num_items))
    
    print("Interaction matrix created successfully.")
    return interactions

def train_model(interactions):
    """使用 LightFM 训练协同过滤模型"""
    print("\nStep 3: Training the LightFM model...")
    
    # 初始化模型，并设置 random_state 以保证结果可复现
    model = LightFM(loss='warp', random_state=42)
    
    # 训练模型
    # epochs=10 表示在整个数据集上迭代10次
    model.fit(interactions, epochs=10, num_threads=4)
    
    print("Model training completed successfully!")
    return model

def main():
    """主执行函数"""
    df, num_users, num_items = load_data()
    interactions = create_interaction_matrix(df, num_users, num_items)
    model = train_model(interactions)
    print("\nBaseline model for Project B has been successfully trained.")

if __name__ == "__main__":
    main()