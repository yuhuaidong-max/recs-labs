# 项目B - 轻量级电影推荐服务

### 项目简介
这是一个基于协同过滤算法的、通过 FastAPI 和 Docker 部署的轻量级电影推荐服务。项目使用经典的 MovieLens 100k 数据集，实现了从离线模型训练到线上 API 推理的全流程，并通过单元测试和健康检查接口保证了服务的可靠性。

### 技术栈
* **语言**: Python 3.9+
* **核心库**: FastAPI, LightFM, Scikit-learn, Pandas, Joblib
* **服务部署**: Uvicorn, Docker
* **测试**: Pytest

### API 接口文档

#### 1. 健康检查
* **URL**: `/health`
* **Method**: `GET`
* **参数**: 无
* **成功返回 (200 OK)**:
  ```json
  {
    "status": "ok"
  }
  ```

#### 2. 获取推荐
* **URL**: `/recommend`
* **Method**: `GET`
* **参数**:
  * `user_id` (integer, required): 需要获取推荐的用户ID。
* **成功返回 (200 OK)**:
  ```json
  {
    "user_id": 1,
    "recommendations": [89, 121, 98, 118, 100, 111, 96, 114, 95, 93]
  }
  ```

### 如何本地运行 (以 Codespaces 为例)

1.  **环境准备**：
    `conda activate recs-labs-env`
2.  **训练模型** (如果模型文件 `lightfm_model.joblib` 不存在):
    `python train.py`
3.  **启动服务**:
    `uvicorn serve:app --reload`
    服务将在 8000 端口启动。

### 如何通过 Docker 运行

1.  **构建镜像**:
    `docker build -t rec-service .`
2.  **运行容器**:
    `docker run -p 8000:8000 --name my-rec-app rec-service`

### 如何运行测试
1.  首先，在一个终端中启动服务 (`uvicorn serve:app`)。
2.  然后，在**另一个**终端中，激活环境并运行：
    `pytest`