import os
import xarray as xr
import numpy as np
import dask
from distributed import Client, LocalCluster
from typing import Dict, Any, Callable

# 从 minian_core 子文件夹中导入核心模块
from .minian_core import preprocessing, motion_correction, initialization, cnmf, utilities


class MinianPipeline:
    def __init__(self, data_manager: 'DataManager', varr: xr.DataArray):
        self.data_manager = data_manager
        self.varr = varr
        self.results = {}  # 存储每个步骤的结果
        self.A = None
        self.C = None
        self.S = None
        self.b = None
        self.f = None
        self.video_files = [] # 存储视频文件名

        # 将原始视频数据存储到结果字典中
        self.results["raw_video"] = self.varr
        print("原始视频数据已加载到管道中。")

        # 启动 Dask 本地集群，用于并行计算
        dask.config.set({"array.slicing.split_large_chunks": False})
        self.cluster = LocalCluster()
        self.client = Client(self.cluster)
        print("Dask 本地集群已启动。")
        
        # 定义每个步骤的默认参数
        self.step_params_map = {
            "step1_1": {"min_size": 10},
            "step1_2": {"method": "median", "wnd_size": 5},
            "step1_3": {"method": "uniform", "wnd": 20},
            "step2": {"npart": 3, "chunk_nfm": None},
            "step3_1": {"wnd_size": 500, "method": "rolling", "stp_size": 200, "nchunk": 100, "max_wnd": 10, "diff_thres": 2},
            "step3_2": {"ks_alpha": 0.05, "pnr_thres": 2.5},
            "step3_3": {"thres": 0.75},
            "step3_4": {},
            "step4_1": {"max_iter": 5, "smoothness": 1},
            "step4_2": {"max_iter": 5, "smoothness": 1},
            "step5": {},
        }

    def run_step(self, step_name: str, params: Dict[str, Any], progress_callback: Callable = None):
        """
        运行给定的步骤，并根据需要更新进度。
        """
        # 伪进度条模拟
        total_steps = 100
        current_progress = 0

        def update_progress(p):
            nonlocal current_progress
            current_progress = int(p)
            if progress_callback:
                progress_callback(current_progress)

        try:
            if step_name == "step1_1":
                varr_processed = preprocessing.remove_glow(self.varr, **params)
                self.results["varr_no_glow"] = varr_processed
                print("步骤1.1: 去除光晕完成。")

            elif step_name == "step1_2":
                if "varr_no_glow" not in self.results: raise KeyError("请先运行步骤1.1。")
                varr_denoised = preprocessing.denoise_movie(self.results["varr_no_glow"], **params)
                self.results["varr_denoised"] = varr_denoised
                print("步骤1.2: 去噪完成。")

            elif step_name == "step1_3":
                if "varr_denoised" not in self.results: raise KeyError("请先运行步骤1.2。")
                varr_subtracted = preprocessing.remove_background(self.results["varr_denoised"], **params)
                self.results["varr_processed"] = varr_subtracted
                print("步骤1.3: 去除背景完成。")

            elif step_name == "step2":
                if "varr_processed" not in self.results: raise KeyError("请先运行步骤1.3。")
                varr_mc = motion_correction.estimate_motion(self.results["varr_processed"], **params)
                self.results["varr_mc"] = varr_mc
                print("步骤2: 运动校正完成。")
            
            elif step_name == "step3_1":
                if "varr_mc" not in self.results: raise KeyError("请先运行步骤2。")
                seeds = initialization.seeds_init(self.results["varr_mc"], **params)
                self.results["seeds"] = seeds
                print("步骤3.1: 种子生成完成。")
            
            elif step_name == "step3_2":
                if "varr_mc" not in self.results or "seeds" not in self.results: raise KeyError("请先运行步骤2和3.1。")
                seeds_pnr_ks = initialization.pnr_refine(self.results["varr_mc"], self.results["seeds"], **params)
                self.results["seeds_pnr_ks"] = seeds_pnr_ks
                print("步骤3.2: PNR和KS精炼完成。")
            
            elif step_name == "step3_3":
                if "seeds_pnr_ks" not in self.results: raise KeyError("请先运行步骤3.2。")
                seeds_merged = initialization.seeds_merge(self.results["seeds_pnr_ks"], **params)
                self.results["seeds_merged"] = seeds_merged
                print("步骤3.3: 合并种子完成。")
            
            elif step_name == "step3_4":
                if "varr_mc" not in self.results or "seeds_merged" not in self.results: raise KeyError("请先运行步骤2和3.3。")
                A_init, C_init = initialization.cnmf_init(self.results["varr_mc"], self.results["seeds_merged"])
                self.results["A"] = A_init
                self.results["C"] = C_init
                print("步骤3.4: CNMF初始化完成。")

            elif step_name == "step4_1":
                if "A" not in self.results or "C" not in self.results: raise KeyError("请先运行步骤3.4。")
                A_new, C_new = cnmf.cnmf_iter(self.results["varr_mc"], self.results["A"], self.results["C"], **params)
                self.results["A"] = A_new
                self.results["C"] = C_new
                print("步骤4.1: 第一次迭代完成。")
            
            elif step_name == "step4_2":
                if "A" not in self.results or "C" not in self.results: raise KeyError("请先运行步骤4.1。")
                A_new, C_new = cnmf.cnmf_iter(self.results["varr_mc"], self.results["A"], self.results["C"], **params)
                self.results["A"] = A_new
                self.results["C"] = C_new
                print("步骤4.2: 第二次迭代完成。")

            elif step_name == "step5":
                if not all(k in self.results for k in ["A", "C"]): raise KeyError("请先运行步骤4.2。")
                # 保存结果的逻辑
                # utilities.save_minian(self.results["A"], ...)
                # ...
                print("步骤5: 结果保存完成。")
            
            else:
                raise ValueError(f"未知的步骤名称: {step_name}")
                
            update_progress(100)
            
        except KeyError as e:
            print(f"错误: {e}")
            update_progress(0)
        except Exception as e:
            print(f"运行步骤 {step_name} 时发生错误: {e}")
            update_progress(0)
    
    def get_step_result(self, result_key: str):
        """获取特定步骤的结果数据"""
        if result_key in self.results:
            return self.results[result_key]
        else:
            return None