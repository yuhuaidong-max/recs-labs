# recs-labs/ctr-avazu/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import time
import os

# --- 1. 数据加载与采样 ---
print("Step 1: Loading and sampling data...")
start_time = time.time()

# 数据路径
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(script_dir, 'train.csv')
# 由于原始数据很大，我们只读取前10万行作为快速实验
N_ROWS = 100000

# 使用pandas加载数据
# 为了节省内存，我们可以指定部分列的数据类型
# 这里我们选择一些有代表性的特征列和目标列'click'
feature_cols = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
target_col = ['click']

try:
    df = pd.read_csv(DATA_FILE, nrows=N_ROWS, usecols=feature_cols + target_col)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE}")
    print("Please ensure you have downloaded the Avazu train.csv and placed it in the 'ctr-avazu' directory.")
    exit() # 如果文件不存在，则退出脚本

load_time = time.time() - start_time
print(f"Data loaded in {load_time:.2f} seconds.")
print("Data sample (first 5 rows):")
print(df.head())
print("\n")


# --- 2. 简单特征工程 ---
print("Step 2: Performing simple feature engineering...")
# 工业界CTR模型特征工程非常复杂，这里我们做一个极简版的示例
# 我们将'hour'字段处理一下，提取出小时信息 (格式是 14102100 -> 2014-10-21 00:00:00)
# Pandas的to_datetime可以智能解析
df['hour_of_day'] = pd.to_datetime(df['hour'], format='%y%m%d%H').dt.hour

# 对于类别特征，逻辑回归需要它们是数值形式。我们使用One-Hot编码。
# 这里为了简化，我们只选择几个基数(cardinality)不那么大的类别特征做One-Hot
categorical_features = ['banner_pos', 'site_category', 'app_category', 'device_type', 'device_conn_type']

# 使用get_dummies进行One-Hot编码
df_onehot = pd.get_dummies(df, columns=categorical_features, dummy_na=False)

print("Feature engineering complete. New features created.")
print(f"Number of features after one-hot encoding: {len(df_onehot.columns) - len(target_col) - len(feature_cols) + len(categorical_features)}")
print("\n")


# --- 3. 准备训练数据 ---
print("Step 3: Preparing data for training...")
    
# 从DataFrame中分离出目标变量y和特征X
y = df_onehot['click']
X = df_onehot.drop(columns=['click', 'hour']) # 从特征矩阵中移除目标列和原始的hour列

# 为确保模型只使用数值型特征，我们筛选出所有数值类型的列
# 这包括了原始的数值特征，以及新生成的One-Hot编码列（它们的值是0或1）
# 'uint8' 是 pd.get_dummies 常用的一种数据类型
X = X.select_dtypes(include=['number', 'uint8'])

# 将数据划分为训练集和测试集 (80%训练, 20%测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print("\n")


# --- 4. 训练逻辑回归模型 ---
print("Step 4: Training a Logistic Regression model...")
train_start_time = time.time()

# 初始化模型
# solver='lbfgs'是常用的优化算法, max_iter确保模型有足够次数来收敛
model = LogisticRegression(solver='lbfgs', max_iter=1000)

# 训练模型
model.fit(X_train, y_train)

train_time = time.time() - train_start_time
print(f"Model trained in {train_time:.2f} seconds.")
print("\n")


# --- 5. 评估模型 ---
print("Step 5: Evaluating the model...")
# 使用模型对测试集进行预测，注意我们获取的是属于类别1（点击）的概率
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 计算AUC和LogLoss
auc = roc_auc_score(y_test, y_pred_proba)
logloss = log_loss(y_test, y_pred_proba)

print("Evaluation complete.")
print("--- Model Performance ---")
print(f"AUC: {auc:.6f}")
print(f"LogLoss: {logloss:.6f}")
print("-------------------------")
print("\n")

print("Script finished successfully!")