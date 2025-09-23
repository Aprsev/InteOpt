import os

# 全局参数配置文件名
GLOBAL_CONFIG_FILE = "global_config.json"

# 数据仓库文件夹名（用于存储库级配置、数据库等）
REPO_FOLDER_NAME = ".minian_repo"

# 数据库文件名
DATABASE_FILE = "minian_data.db"

# 步骤名称
STEP_NAMES = {
    "step0": "步骤0: 原始视频预览",
    "step1_1": "步骤1.1: 去除光晕",
    "step1_2": "步骤1.2: 去噪",
    "step1_3": "步骤1.3: 去除背景",
    "step2": "步骤2: 运动校正",
    "step3_1": "步骤3.1: 种子生成",
    "step3_2": "步骤3.2: PNR和KS精炼",
    "step3_3": "步骤3.3: 合并种子",
    "step3_4": "步骤3.4: CNMF初始化",
    "step4_1": "步骤4.1: 第一次迭代",
    "step4_2": "步骤4.2: 第二次迭代",
    "step5": "步骤5: 结果保存",
}