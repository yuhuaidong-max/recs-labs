# 项目A - CTR (点击率) 预估离线流水线

### 项目简介
本项目旨在利用 Kaggle 的 Avazu CTR Prediction 竞赛数据集，构建一个完整的广告点击率（Click-Through Rate, CTR）预估离线流水线。通过对比不同的机器学习模型和特征工程策略，我们旨在得到一个性能优异的基线模型，并展示其在实际场景下的推理能力。

### 技术栈
* **语言**: Python 3.9+
* **核心库**: Pandas, Scikit-learn, XGBoost, Joblib

### 项目结构
本项目包含两个核心脚本：
-   `train.py`: 负责整个模型的离线训练和评估流程。它包含了数据加载、特征工程、模型训练与性能对比的完整逻辑。
-   `infer.py`: 负责模型的线上推理。它加载 `train.py` 训练好的模型，并对新的模拟数据进行点击率预测。
-   `requirements.txt`: 记录了项目所有的依赖库及其版本，便于环境复现。
-   `results.md`: 记录了不同实验的量化对比结果。
-   `ctr-avazu/`: 包含本次项目的主要代码文件和原始数据。

### 核心流程
1.  **数据加载与预处理**: 从 `train.csv` 文件中加载原始数据，并进行基础预处理（如时间特征提取）。
2.  **特征工程**: 
    -   对低基数的类别特征（如 `banner_pos`）进行 One-Hot 编码。
    -   对高基数的类别特征（如 `site_id`, `app_id`）采用**特征哈希 (Feature Hashing)** 技术处理，以避免维度灾难。
    -   实践了**负采样 (Negative Sampling)** 技术，以应对数据集的极端不平衡问题。
3.  **模型训练与评估**: 
    -   我们训练并评估了两种模型：逻辑回归（LR）和 XGBoost。
    -   在不同特征集（原始特征 vs. 哈希特征）上对模型性能进行了量化对比。
4.  **模型推理**: 将训练好的最佳模型保存，并使用独立的 `infer.py` 脚本进行线上推理测试。

### 实验结果对比
| 模型 (Model) | 特征集 (Feature Set) | AUC | LogLoss |
| :--- | :--- | :--- | :--- |
| Logistic Regression | Original | 0.646152 | 0.443348 |
| XGBoost | Original | 0.701039 | 0.424913 |
| Logistic Regression | Hashed | 0.657101 | 0.437521 |
| XGBoost | Hashed | **0.719450** | **0.417715** |
| XGBoost | Hashed + Sampled | 0.712836 | 0.542072 |

### 如何运行本项目
1.  **克隆仓库**：
    ```bash
    git clone [https://github.com/yuhuaidong-max/recs-labs.git](https://github.com/yuhuaidong-max/recs-labs.git)
    cd recs-labs
    ```
2.  **环境准备**：
    ```bash
    # 创建并激活虚拟环境
    python -m venv env
    # Mac/Linux:
    source env/bin/activate
    # Windows:
    .\env\Scripts\activate

    # 安装依赖
    pip install -r requirements.txt
    ```
3.  **下载数据**：
    请从 Kaggle [Avazu CTR Prediction](https://www.kaggle.com/competitions/avazu-ctr-prediction/data) 竞赛页面下载 `train.gz` 文件，解压后将 `train.csv` 放入 `ctr-avazu/` 目录下。

4.  **训练模型**：
    (此过程会生成 `ctr-avazu/xgb_model.joblib` 和 `ctr-avazu/ohe_columns.joblib` 文件)
    ```bash
    python ctr-avazu/train.py
    ```
5.  **进行推理**：
    ```bash
    python ctr-avazu/infer.py
    ```