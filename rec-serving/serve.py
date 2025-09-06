from fastapi import FastAPI
import numpy as np
import joblib
import os

# --- 1. 初始化和加载模型 ---

# 创建 FastAPI 应用实例
app = FastAPI(title="轻量推荐服务", description="一个基于 LightFM 模型的电影推荐 API")

# 定义模型文件路径
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(script_dir, 'lightfm_model.joblib')

# 在应用启动时加载模型（只执行一次）
try:
    model_data = joblib.load(MODEL_FILE)
    model = model_data['model']
    num_users = model_data['num_users']
    num_items = model_data['num_items']
    interaction_matrix = model_data['interaction_matrix']
    print("模型和数据加载成功！")
except FileNotFoundError:
    print("错误：找不到模型文件。请先运行 train.py 生成模型。")
    model = None

# --- 2. 编写预测函数 ---

def get_recommendations(user_id: int, top_n: int = 10):
    """为指定用户生成 Top-N 推荐"""
    if model is None:
        return {"error": "模型未加载，无法提供推荐。"}

    # LightFM 内部使用 0-indexed ID
    internal_user_id = user_id - 1

    # 获取用户已经交互过的 item 列表
    # interaction_matrix.tocsr() 转换为更高效的行访问格式
    known_positives = interaction_matrix.tocsr()[internal_user_id].indices

    # 生成所有 item 的预测分数
    # np.arange(num_items) 生成一个从 0 到 num_items-1 的数组
    scores = model.predict(internal_user_id, np.arange(num_items))
    
    # 过滤掉用户已经交互过的 item
    # 我们给已知 item 的分数设置一个极低的值，确保它们不会被推荐
    scores[known_positives] = -np.inf

    # 获取分数最高的 Top-N item 的索引 (0-indexed)
    top_items_internal = np.argsort(-scores)[:top_n]
    
    # 将内部索引转换回原始的 MovieLens item_id (1-indexed)
    top_items_original = [int(item_id + 1) for item_id in top_items_internal]
    
    return top_items_original

# --- 3. 创建 API 接口 ---

@app.get("/recommend", summary="获取电影推荐", description="为指定用户ID返回Top-N个电影推荐列表")
def recommend_for_user(user_id: int):
    """
    接收一个 user_id，返回推荐的 item_id 列表。
    访问示例: /recommend?user_id=1
    """
    recommendations = get_recommendations(user_id)
    
    if "error" in recommendations:
        return recommendations
    
    return {"user_id": user_id, "recommendations": recommendations}

# --- 4. 根路径欢迎信息 ---
@app.get("/", summary="服务欢迎页")
def read_root():
    return {"message": "欢迎使用轻量推荐服务！请访问 /docs 查看 API 文档。"}