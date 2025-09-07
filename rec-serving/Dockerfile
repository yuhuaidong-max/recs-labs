# 1. 使用官方的Python 3.9作为基础镜像
FROM python:3.9-slim

# --- 新增步骤：安装编译工具 ---
# 在运行pip install之前，先用apt-get更新包列表并安装build-essential
# build-essential 包含了gcc等所有编译C/C++代码所需的工具
RUN apt-get update && apt-get install -y build-essential
# --- 新增结束 ---

# 2. 将工作目录设置为/app
WORKDIR /app

# 3. 将 rec-serving 目录下的所有文件复制到镜像的/app目录下
COPY . .

# 4. 在镜像中安装项目依赖
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple fastapi uvicorn "lightfm>=1.17" "scikit-learn>=1.0" "joblib>=1.0" "pandas>=2.0" "numpy>=1.20"

# 5. 容器启动时要执行的命令：启动FastAPI服务
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]