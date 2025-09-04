import pandas as pd
import numpy as np
import joblib
import os

from sklearn.feature_extraction import FeatureHasher
from scipy.sparse import hstack, csr_matrix

# 定义与 train.py 中一致的特征列表
low_cardinality_categorical = ['banner_pos', 'site_category', 'app_category', 
                               'device_type', 'device_conn_type']
high_cardinality_categorical = ['site_id', 'site_domain', 'app_id', 'app_domain',
                                'device_id', 'device_ip', 'device_model']
numerical_features = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'hour_of_day']

# 文件路径
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(script_dir, 'xgb_model.joblib')
OHE_COLUMNS_FILE = os.path.join(script_dir, 'ohe_columns.joblib')

def preprocess_new_data(new_data, with_hashing=True, hash_features=2**14):
    """
    预处理新数据，使其与训练时的特征格式一致。
    """
    # 原始的hour列转换为hour_of_day
    new_data['hour_of_day'] = pd.to_datetime(new_data['hour'], format='%y%m%d%H').dt.hour
    new_data = new_data.drop(columns=['hour'])

    if with_hashing:
        hasher = FeatureHasher(n_features=hash_features, input_type='string')
        high_card_data = new_data[high_cardinality_categorical].astype(str).to_dict('records')
        hashed_features = hasher.fit_transform(high_card_data)
        
        # --- 核心修正点 ---
        # 1. 对新数据的低基数特征进行One-Hot编码
        df_low_card = pd.get_dummies(new_data[low_cardinality_categorical], columns=low_cardinality_categorical)
        
        # 2. 加载训练时保存的One-Hot列名
        ohe_columns = joblib.load(OHE_COLUMNS_FILE)
        
        # 3. 使用.reindex()强制对齐列，缺少的列用0填充
        df_low_card_aligned = df_low_card.reindex(columns=ohe_columns, fill_value=0)
        # --- 修正结束 ---

        X = hstack([
            csr_matrix(new_data[numerical_features].values),
            csr_matrix(df_low_card_aligned.astype(int).values), # 使用对齐后的数据
            hashed_features
        ]).tocsr()
    else:
        # 非哈希的情况也需要修正，但我们当前模型用的是哈希，故简化
        raise NotImplementedError("Inference for non-hashing model is not implemented in this version.")
    
    return X

def main():
    """主执行函数"""
    # 模拟一条新的广告曝光数据
    new_ad_data = pd.DataFrame([
        {
            'hour': 14102210, 'C1': 1005, 'banner_pos': 0, 'site_category': '28905ebd', 
            'app_category': '07d7df22', 'device_type': 1, 'device_conn_type': 2,
            'C14': 15706, 'C15': 320, 'C16': 50, 'C17': 1722, 'C18': 0, 'C19': 35, 
            'C20': 100084, 'C21': 79, 'site_id': '1fbe01fe', 'site_domain': 'f3845767',
            'app_id': 'ecad2386', 'app_domain': '7801e8d9', 'device_id': 'a99f214a',
            'device_ip': '91206129', 'device_model': 'd73a7266'
        },
        {
            'hour': 14102211, 'C1': 1002, 'banner_pos': 1, 'site_category': 'f6ab5888',
            'app_category': '471804f5', 'device_type': 1, 'device_conn_type': 0,
            'C14': 20058, 'C15': 320, 'C16': 50, 'C17': 2316, 'C18': 0, 'C19': 282,
            'C20': 100156, 'C21': 48, 'site_id': 'e151e245', 'site_domain': '7e0520a7',
            'app_id': 'ecad2386', 'app_domain': '7801e8d9', 'device_id': 'a99f214a',
            'device_ip': '91206129', 'device_model': 'd73a7266'
        }
    ])

    print("Step 1: Loading trained model and feature columns...")
    try:
        model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print("Error: Model file not found. Please ensure you have run train.py to generate it.")
        return

    print("Step 2: Preprocessing new data...")
    X_new = preprocess_new_data(new_ad_data)

    print("Step 3: Making prediction...")
    prediction_proba = model.predict_proba(X_new)[:, 1]
    
    print("\n--- CTR Prediction Results ---")
    print("Feature Set:")
    print(new_ad_data.head())
    print("\nPrediction Probabilities:")
    print(prediction_proba)
    print("------------------------------")
    
if __name__ == "__main__":
    main()