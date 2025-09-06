import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from lightfm import LightFM
import os
import joblib # 导入 joblib 用于保存模型

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
        print("Please ensure you have downloaded the ml-100k dataset.")
        exit()

    print("Data loaded successfully.")
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()
    print(f"Number of unique users: {num_users}, Number of unique items: {num_items}")
    
    return df, num_users, num_items

def create_interaction_matrix(df, num_users, num_items):
    """根据 DataFrame 创建用户-物品交互稀疏矩阵"""
    print("\nStep 2: Creating user-item interaction matrix...")
    df['user_id'] = df['user_id'] - 1
    df['item_id'] = df['item_id'] - 1

    interactions = coo_matrix((df['rating'], (df['user_id'], df['item_id'])), 
                              shape=(num_users, num_items))
    
    print("Interaction matrix created successfully.")
    return interactions

def train_and_save_model(interactions, num_users, num_items):
    """使用 LightFM 训练模型并将其保存到文件"""
    print("\nStep 3: Training the LightFM model...")
    model = LightFM(loss='warp', random_state=42)
    model.fit(interactions, epochs=10, num_threads=4)
    print("Model training completed successfully!")

    # --- 新增代码：保存模型和相关数据 ---
    print("\nStep 4: Saving the model and necessary data...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建一个字典来保存所有需要的东西
    model_data = {
        'model': model,
        'num_users': num_users,
        'num_items': num_items,
        'interaction_matrix': interactions # 我们也保存交互矩阵，以便后续过滤已看过的电影
    }
    
    # 使用 joblib 保存
    joblib.dump(model_data, os.path.join(script_dir, 'lightfm_model.joblib'))
    print(f"Model and data saved to 'rec-serving/lightfm_model.joblib'")
    # --- 新增结束 ---

def main():
    """主执行函数"""
    df, num_users, num_items = load_data()
    interactions = create_interaction_matrix(df, num_users, num_items)
    train_and_save_model(interactions, num_users, num_items)
    print("\nTraining script for Project B has been successfully executed.")

if __name__ == "__main__":
    main()