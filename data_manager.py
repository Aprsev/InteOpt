import json
import os
import sqlite3
from typing import Dict, Any

from .config import REPO_FOLDER_NAME, DATABASE_FILE, GLOBAL_CONFIG_FILE


class DataManager:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.repo_folder = os.path.join(self.repo_path, REPO_FOLDER_NAME)
        self.db_path = os.path.join(self.repo_folder, DATABASE_FILE)
        self.conn = None

    def initialize_repo(self):
        """初始化数据仓库，创建文件夹和数据库"""
        if not os.path.exists(self.repo_folder):
            os.makedirs(self.repo_folder)
            print(f"数据仓库已创建: {self.repo_folder}")

        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        cursor.execute('''
           CREATE TABLE IF NOT EXISTS parameters (
               step_name TEXT PRIMARY KEY,
               params TEXT
           )
        ''')
        self.conn.commit()
        print(f"数据库已创建: {self.db_path}")

    def load_params(self, mode: str, step_name: str) -> Dict[str, Any]:
        """根据模式和步骤加载参数"""
        if mode == "debug":
            # 调试模式，从数据库加载上一次保存的参数
            if self.conn:
                cursor = self.conn.cursor()
                cursor.execute("SELECT params FROM parameters WHERE step_name=?", (step_name,))
                result = cursor.fetchone()
                if result:
                    return json.loads(result[0])
        elif mode == "run":
            # 运行模式，从库配置文件或全局配置加载
            # 优先从库配置文件加载
            repo_config_path = os.path.join(self.repo_folder, f"{step_name}_config.json")
            if os.path.exists(repo_config_path):
                with open(repo_config_path, 'r') as f:
                    return json.load(f)
            # 如果没有库配置文件，加载全局配置
            elif os.path.exists(GLOBAL_CONFIG_FILE):
                with open(GLOBAL_CONFIG_FILE, 'r') as f:
                    return json.load(f)
        return {}  # 返回空字典表示没有找到

    def save_params(self, step_name: str, params: Dict[str, Any]):
        """将参数保存到数据库（调试模式）或配置文件（调试完成后）"""
        if self.conn:
            cursor = self.conn.cursor()
            params_str = json.dumps(params)
            cursor.execute(
                "INSERT OR REPLACE INTO parameters (step_name, params) VALUES (?, ?)",
                (step_name, params_str)
            )
            self.conn.commit()

    def save_global_config(self, step_name: str, params: Dict[str, Any]):
        """将参数保存为全局配置文件"""
        with open(GLOBAL_CONFIG_FILE, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"全局配置文件已保存: {GLOBAL_CONFIG_FILE}")

    def close(self):
        if self.conn:
            self.conn.close()