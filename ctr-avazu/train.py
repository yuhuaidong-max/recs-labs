import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.feature_extraction import FeatureHasher
from scipy.sparse import hstack, csr_matrix
import time
import os

def load_and_preprocess_data(nrows, with_hashing=False, hash_features=10000):
    """
    加载并对数据进行预处理。
    :param nrows: 读取的数据行数
    :param with_hashing: 是否对高基数特征进行哈希处理
    :param hash_features: 哈希特征的数量
    :return: 处理后的特征矩阵X和目标向量y
    """
    print(f"\nStep 1: Loading and preprocessing data... (Hashing={'Yes' if with_hashing else 'No'})")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_FILE = os.path.join(script_dir, 'train.csv')
    
    # 定义不同类型的特征列
    target_col = 'click'
    low_cardinality_categorical = ['banner_pos', 'site_category', 'app_category', 
                                   'device_type', 'device_conn_type']
    high_cardinality_categorical = ['site_id', 'site_domain', 'app_id', 'app_domain',
                                    'device_id', 'device_ip', 'device_model']
    # C* 特征也是类别型，但由于是数值，我们暂时作为普通数值特征处理
    numerical_features = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
    
    all_feature_cols = low_cardinality_categorical + high_cardinality_categorical + numerical_features + ['hour']
    
    df = pd.read_csv(DATA_FILE, nrows=nrows, usecols=all_feature_cols + [target_col])
    
    # 目标变量
    y = df[target_col]
    df = df.drop(columns=[target_col]) # 从df中移除y，剩下的全是特征

    # 特征工程: 时间处理
    df['hour_of_day'] = pd.to_datetime(df['hour'], format='%y%m%d%H').dt.hour
    numerical_features.append('hour_of_day')
    df = df.drop(columns=['hour'])
    
    if with_hashing:
        # 使用特征哈希处理高基数特征
        hasher = FeatureHasher(n_features=hash_features, input_type='string')
        # 将高基数特征列转换为字符串，然后按行转换为字典列表
        high_card_data = df[high_cardinality_categorical].astype(str).to_dict('records')
        hashed_features = hasher.fit_transform(high_card_data)
        
        # 对低基数特征进行One-Hot编码
        df_low_card = pd.get_dummies(df[low_cardinality_categorical], columns=low_cardinality_categorical)
        
        # 组合所有特征
        # 注意: FeatureHasher和get_dummies都可能产生稀疏矩阵，所以我们用hstack
        X = hstack([
            csr_matrix(df[numerical_features].values),
            csr_matrix(df_low_card.values),
            hashed_features
        ]).tocsr() # tocsr() 确保最终是csr格式的稀疏矩阵
        
    else:
        # 原始方法：只对低基数特征进行One-Hot编码
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
    HASH_FEATURES = 2**14 # 哈希到16384个特征维度

    results = []

    # --- 实验一：在原始特征上对比 LR 和 XGBoost ---
    print("="*20 + " Experiment 1: LR vs. XGBoost on Original Features " + "="*20)
    X_orig, y_orig = load_and_preprocess_data(nrows=N_ROWS, with_hashing=False)
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)
    
    lr_model = LogisticRegression(solver='lbfgs', max_iter=2000)
    lr_auc, lr_logloss = train_and_evaluate(lr_model, X_train_orig, y_train_orig, X_test_orig, y_test_orig)
    results.append(['Logistic Regression', 'Original', lr_auc, lr_logloss])

    xgb_model = xgb.XGBClassifier(eval_metric='logloss')
    xgb_auc, xgb_logloss = train_and_evaluate(xgb_model, X_train_orig, y_train_orig, X_test_orig, y_test_orig)
    results.append(['XGBoost', 'Original', xgb_auc, xgb_logloss])

    # --- 实验二：在哈希特征上对比 LR 和 XGBoost ---
    print("="*20 + " Experiment 2: LR vs. XGBoost on Hashed Features " + "="*20)
    X_hashed, y_hashed = load_and_preprocess_data(nrows=N_ROWS, with_hashing=True, hash_features=HASH_FEATURES)
    X_train_hashed, X_test_hashed, y_train_hashed, y_test_hashed = train_test_split(X_hashed, y_hashed, test_size=0.2, random_state=42)
    
    lr_model_hashed = LogisticRegression(solver='lbfgs', max_iter=2000)
    lr_auc_hashed, lr_logloss_hashed = train_and_evaluate(lr_model_hashed, X_train_hashed, y_train_hashed, X_test_hashed, y_test_hashed)
    results.append(['Logistic Regression', 'Hashed', lr_auc_hashed, lr_logloss_hashed])

    xgb_model_hashed = xgb.XGBClassifier(eval_metric='logloss')
    xgb_auc_hashed, xgb_logloss_hashed = train_and_evaluate(xgb_model_hashed, X_train_hashed, y_train_hashed, X_test_hashed, y_test_hashed)
    results.append(['XGBoost', 'Hashed', xgb_auc_hashed, xgb_logloss_hashed])

    # --- 打印最终结果对比 ---
    results_df = pd.DataFrame(results, columns=['Model', 'Feature Set', 'AUC', 'LogLoss'])
    print("\n" + "="*20 + " Final Results Summary " + "="*20)
    print(results_df)
    print("="*60 + "\n")

if __name__ == "__main__":
    main()