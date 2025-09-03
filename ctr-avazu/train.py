import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import time
import os

def load_and_preprocess_data(nrows):
    """加载并对数据进行基础预处理"""
    print("Step 1: Loading and preprocessing data...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_FILE = os.path.join(script_dir, 'train.csv')
    
    feature_cols = ['hour', 'C1', 'banner_pos', 'site_category', 'app_category', 
                    'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 
                    'C18', 'C19', 'C20', 'C21']
    target_col = 'click'
    
    df = pd.read_csv(DATA_FILE, nrows=nrows, usecols=feature_cols + [target_col])
    
    # 特征工程: 时间处理 + One-Hot编码
    df['hour_of_day'] = pd.to_datetime(df['hour'], format='%y%m%d%H').dt.hour
    
    categorical_features = ['banner_pos', 'site_category', 'app_category', 
                            'device_type', 'device_conn_type']
    df_onehot = pd.get_dummies(df, columns=categorical_features, dummy_na=False)
    
    return df_onehot

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
    N_ROWS = 100000  # 我们仍然使用10万行数据进行快速实验
    
    # --- 实验一：在原始特征上对比 LR 和 XGBoost ---
    print("="*20 + " Experiment 1: LR vs. XGBoost on Original Features " + "="*20)
    
    # 1. 加载和预处理数据
    df_processed = load_and_preprocess_data(nrows=N_ROWS)
    
    # 2. 准备训练和测试数据
    y = df_processed['click']
    X = df_processed.drop(columns=['click', 'hour'])
    X = X.select_dtypes(include=['number', 'uint8'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. 训练和评估逻辑回归模型 (D1的模型)
    # 注意: 我们将max_iter增加到2000来解决D1的收敛警告
    lr_model = LogisticRegression(solver='lbfgs', max_iter=2000)
    lr_auc, lr_logloss = train_and_evaluate(lr_model, X_train, y_train, X_test, y_test)
    
    # 4. 训练和评估XGBoost模型
    xgb_model = xgb.XGBClassifier(eval_metric='logloss')
    xgb_auc, xgb_logloss = train_and_evaluate(xgb_model, X_train, y_train, X_test, y_test)
    
    # 5. 打印结果对比
    print("\n" + "="*20 + " Results for Experiment 1 " + "="*20)
    print("| Model              | AUC      | LogLoss  |")
    print("|--------------------|----------|----------|")
    print(f"| Logistic Regression| {lr_auc:.6f} | {lr_logloss:.6f} |")
    print(f"| XGBoost            | {xgb_auc:.6f} | {xgb_logloss:.6f} |")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()