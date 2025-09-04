import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.feature_extraction import FeatureHasher
from scipy.sparse import hstack, csr_matrix
import time
import os
import joblib

def load_and_preprocess_data(nrows, with_hashing=False, hash_features=10000, with_negative_sampling=False):
    """
    加载并对数据进行预处理。
    :param nrows: 读取的数据行数
    :param with_hashing: 是否对高基数特征进行哈希处理
    :param hash_features: 哈希特征的数量
    :param with_negative_sampling: 是否进行负采样
    :return: 处理后的特征矩阵X和目标向量y
    """
    print(f"\nStep 1: Loading and preprocessing data... (Hashing={'Yes' if with_hashing else 'No'}, Negative Sampling={'Yes' if with_negative_sampling else 'No'})")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_FILE = os.path.join(script_dir, 'train.csv')
    
    # 定义不同类型的特征列
    target_col = 'click'
    low_cardinality_categorical = ['banner_pos', 'site_category', 'app_category', 
                                   'device_type', 'device_conn_type']
    high_cardinality_categorical = ['site_id', 'site_domain', 'app_id', 'app_domain',
                                    'device_id', 'device_ip', 'device_model']
    numerical_features = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
    
    all_feature_cols = low_cardinality_categorical + high_cardinality_categorical + numerical_features + ['hour']
    
    df = pd.read_csv(DATA_FILE, nrows=nrows, usecols=all_feature_cols + [target_col])
    
    # 实践论文思想：负采样
    if with_negative_sampling:
        df_pos = df[df['click'] == 1]
        df_neg = df[df['click'] == 0].sample(frac=0.5, random_state=42)
        df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  - 负采样完成，新数据集大小为: {len(df)} 行")

    # 目标变量
    y = df[target_col]
    df = df.drop(columns=[target_col])

    # 特征工程: 时间处理
    df['hour_of_day'] = pd.to_datetime(df['hour'], format='%y%m%d%H').dt.hour
    numerical_features.append('hour_of_day')
    df = df.drop(columns=['hour'])
    
    if with_hashing:
        hasher = FeatureHasher(n_features=hash_features, input_type='string')
        high_card_data = df[high_cardinality_categorical].astype(str).to_dict('records')
        hashed_features = hasher.fit_transform(high_card_data)
        
        df_low_card = pd.get_dummies(df[low_cardinality_categorical], columns=low_cardinality_categorical)
        
        X = hstack([
            csr_matrix(df[numerical_features].values),
            csr_matrix(df_low_card.values),
            hashed_features
        ]).tocsr()
    else:
        df_onehot = pd.get_dummies(df, columns=low_cardinality_categorical)
        X = df_onehot.select_dtypes(include=['number', 'uint8'])

    return X, y

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """训练模型并返回评估指标"""
    print(f"--- Training model: {model.__class__.__name__} ---")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)
    
    elapsed_time = time.time() - start_time
    print(f"Training and evaluation finished in {elapsed_time:.2f} seconds.")
    
    return auc, logloss

def main():
    """主执行函数"""
    N_ROWS = 100000
    HASH_FEATURES = 2**14

    # --- 准备工作：生成并保存One-Hot编码的列名，以供infer.py使用 ---
    print("Preparing OHE columns for inference consistency...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_FILE = os.path.join(script_dir, 'train.csv')
    df_sample = pd.read_csv(DATA_FILE, nrows=N_ROWS) # 加载数据以获取所有可能的类别
    low_cardinality_categorical = ['banner_pos', 'site_category', 'app_category', 'device_type', 'device_conn_type']
    df_ohe_template = pd.get_dummies(df_sample[low_cardinality_categorical], columns=low_cardinality_categorical)
    joblib.dump(df_ohe_template.columns.tolist(), os.path.join(script_dir, 'ohe_columns.joblib'))
    print("OHE columns saved.")

    results = []

    # --- 实验一：原始特征对比 LR vs. XGBoost ---
    print("="*20 + " Experiment 1: LR vs. XGBoost on Original Features " + "="*20)
    X_orig, y_orig = load_and_preprocess_data(nrows=N_ROWS, with_hashing=False)
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)
    
    lr_model = LogisticRegression(solver='lbfgs', max_iter=2000)
    lr_auc, lr_logloss = train_and_evaluate(lr_model, X_train_orig, y_train_orig, X_test_orig, y_test_orig)
    results.append(['Logistic Regression', 'Original', lr_auc, lr_logloss])

    xgb_model = xgb.XGBClassifier(eval_metric='logloss')
    xgb_auc, xgb_logloss = train_and_evaluate(xgb_model, X_train_orig, y_train_orig, X_test_orig, y_test_orig)
    results.append(['XGBoost', 'Original', xgb_auc, xgb_logloss])

    # --- 实验二：哈希特征对比 LR vs. XGBoost ---
    print("="*20 + " Experiment 2: LR vs. XGBoost on Hashed Features " + "="*20)
    X_hashed, y_hashed = load_and_preprocess_data(nrows=N_ROWS, with_hashing=True, hash_features=HASH_FEATURES)
    X_train_hashed, X_test_hashed, y_train_hashed, y_test_hashed = train_test_split(X_hashed, y_hashed, test_size=0.2, random_state=42)
    
    lr_model_hashed = LogisticRegression(solver='lbfgs', max_iter=2000)
    lr_auc_hashed, lr_logloss_hashed = train_and_evaluate(lr_model_hashed, X_train_hashed, y_train_hashed, X_test_hashed, y_test_hashed)
    results.append(['Logistic Regression', 'Hashed', lr_auc_hashed, lr_logloss_hashed])

    xgb_model_hashed = xgb.XGBClassifier(eval_metric='logloss')
    xgb_auc_hashed, xgb_logloss_hashed = train_and_evaluate(xgb_model_hashed, X_train_hashed, y_train_hashed, X_test_hashed, y_test_hashed)
    results.append(['XGBoost', 'Hashed', xgb_auc_hashed, xgb_logloss_hashed])

    # --- 实验三：负采样对比（在最佳模型上） ---
    print("="*20 + " Experiment 3: Negative Sampling with XGBoost on Hashed Features " + "="*20)
    X_sampled, y_sampled = load_and_preprocess_data(nrows=N_ROWS, with_hashing=True, hash_features=HASH_FEATURES, with_negative_sampling=True)
    X_train_sampled, X_test_sampled, y_train_sampled, y_test_sampled = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)
    
    xgb_model_sampled = xgb.XGBClassifier(eval_metric='logloss')
    xgb_auc_sampled, xgb_logloss_sampled = train_and_evaluate(xgb_model_sampled, X_train_sampled, y_train_sampled, X_test_sampled, y_test_sampled)
    results.append(['XGBoost', 'Hashed + Sampled', xgb_auc_sampled, xgb_logloss_sampled])

    # --- 模型保存 ---
    joblib.dump(xgb_model_hashed, 'ctr-avazu/xgb_model.joblib')
    print("\nBest model (XGBoost with hashed features) saved to ctr-avazu/xgb_model.joblib")

    # --- 打印最终结果 ---
    results_df = pd.DataFrame(results, columns=['Model', 'Feature Set', 'AUC', 'LogLoss'])
    print("\n" + "="*20 + " Final Results Summary " + "="*20)
    print(results_df)
    print("="*60 + "\n")

if __name__ == "__main__":
    main()