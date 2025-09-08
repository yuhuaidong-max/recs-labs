from fastapi import FastAPI, Request
import numpy as np
import joblib
import os
import time
import logging

# --- 0. 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. 初始化和加载模型 ---
app = FastAPI(title="轻量推荐服务", description="一个基于 LightFM 模型的电影推荐 API")

script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(script_dir, 'lightfm_model.joblib')

# 使用 @app.on_event("startup") 装饰器来在应用启动时加载模型
@app.on_event("startup")
def load_model():
    global model_data, model, num_users, num_items, interaction_matrix
    try:
        model_data = joblib.load(MODEL_FILE)
        model = model_data['model']
        num_users = model_data['num_users']
        num_items = model_data['num_items']
        interaction_matrix = model_data['interaction_matrix']
        logger.info("模型和数据加载成功！")
    except FileNotFoundError:
        logger.error("错误：找不到模型文件。请先运行 train.py 生成模型。")
        model = None

# --- 2. 编写预测函数 ---
def get_recommendations(user_id: int, top_n: int = 10):
    """为指定用户生成 Top-N 推荐"""
    if model is None:
        return {"error": "模型未加载，无法提供推荐。"}

    internal_user_id = user_id - 1
    known_positives = interaction_matrix.tocsr()[internal_user_id].indices
    scores = model.predict(internal_user_id, np.arange(num_items))
    scores[known_positives] = -np.inf
    top_items_internal = np.argsort(-scores)[:top_n]
    top_items_original = [int(item_id + 1) for item_id in top_items_internal]
    
    return top_items_original

# --- 3. 创建 API 接口 ---

# 新增：健康检查接口
@app.get("/health", summary="服务健康检查")
def health_check():
    """
    一个简单的接口，返回200 OK状态，用于监控服务是否存活。
    """
    return {"status": "ok"}

@app.get("/recommend", summary="获取电影推荐", description="为指定用户ID返回Top-N个电影推荐列表")
def recommend_for_user(user_id: int, request: Request):
    """
    接收一个 user_id，返回推荐的 item_id 列表。
    """
    start_time = time.time()
    logger.info(f"收到用户 {user_id} 的推荐请求，来源 IP: {request.client.host}")
    
    recommendations = get_recommendations(user_id)
    
    if "error" in recommendations:
        return recommendations
    
    duration = time.time() - start_time
    logger.info(f"为用户 {user_id} 的请求处理完毕，耗时: {duration:.4f} 秒")
    
    return {"user_id": user_id, "recommendations": recommendations}

# 根路径欢迎信息
@app.get("/", summary="服务欢迎页")
def read_root():
    return {"message": "欢迎使用轻量推荐服务！请访问 /docs 查看 API 文档。"}