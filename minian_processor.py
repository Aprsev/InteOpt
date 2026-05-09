# D:\Desktop\ZJU\SRTP\ui_v1\minian_processor.py

import numpy as np
import xarray as xr
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Union, List
import json
import os
import dask.array as da
import dask.dataframe as dd
import matplotlib.pyplot as plt
import cv2
import traceback

# ====================================================================
# 修正后的导入块：严格根据您提供的函数列表和错误路径 (minian_core)
# ====================================================================

from minian_core.cnmf import ( 
    compute_AtC,
    compute_trace,
    get_noise_fft,
    smooth_sig,
    unit_merge,
    update_spatial, # ✅ CNMF 空间更新
    update_temporal, # ✅ CNMF 时间更新
    update_background,
)
from minian_core.initialization import (
    gmm_refine,
    initA,       # ✅ 空间初始化函数
    initC,       # ✅ 时间初始化函数
    intensity_refine,
    ks_refine,
    pnr_refine,
    seeds_init,
    seeds_merge, # ✅ 修正：用于合并种子点的正确函数名
)
from minian_core.motion_correction import apply_transform, estimate_motion,apply_shifts
from minian_core.preprocessing import denoise, remove_background,remove_glow
from minian_core.utilities import (
    TaskAnnotation,
    get_optimal_chk,
    load_videos,
    open_minian,
    save_minian,
    # ... 其他 utilities 函数
)
from minian_core.visualization import (
    create_cnmf_update_plot, 
    create_spatial_update_plot, 
    create_temporal_matrix_plot, 
    create_merge_matrix_plot,
    create_spatial_exploration_plot,          
    create_spatial_exploration_compare_plot   
)


class MinianProcessor:
    """
    Minian 处理流程的封装类。
    请将以下方法替换/补充到您现有类中，以确保函数调用正确。
    """
    
    def __init__(self, video_folder: str, config_path: str, repo_dir: str = None):
        self.video_folder = video_folder 
        self.config_path = config_path 
        self.dpath = video_folder 
        
        self.repo_dir = repo_dir or os.path.dirname(os.path.abspath(__file__))
        self.fps = 20.0
        self.log_output = [] 
        self.step_statuses = {}
        self.steps_results = {}
        self.exploration_results = {} # save the result of exploration
        self.exploration_state = {}   # save the state of exploration (e.g., current index in parameter list)
        
        self.data_path = os.path.join(self.dpath, "minian_visual_cache")
        
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f) 
            # 可以在此处添加日志或打印信息
            print(f"配置文件从 {config_path} 加载成功。")
        except FileNotFoundError:
            print(f"错误: 配置文件未找到于 {config_path}")
            self.config = {}
        except json.JSONDecodeError:
            print(f"错误: 配置文件 {config_path} 格式错误 (非有效 JSON)。")
            self.config = {}
        except Exception as e:
            print(f"加载配置文件时发生其他错误: {e}")
            self.config = {}
        
        # 🔴 关键修正 4：初始化数据仓库
        self.data_repo = {} 
        
    def get_step_params(self, step_name: str) -> Dict[str, Any]:
        # 您的获取参数方法代码...
        print(f"DEBUG: 获取步骤 '{step_name}' 的参数。{self.config.get(step_name, {})}")
        return self.config.get(step_name, {})

    def _load_data_from_repo(self, key: str) -> Any:
        """从仓库加载数据"""
        return self.data_repo.get(key)
        
    def _save_data_to_repo(self, data: Any, key: str) -> Any:
        """保存数据到仓库"""
        self.data_repo[key] = data
        return data
        
    def _update_config_param(self, param_name, value):
        """
        更新配置文件中的参数
        """
        if hasattr(self, 'config'):
            self.config[param_name] = value
            # 保存更新后的配置
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                    
    def get_video_fps(self):
        """UI 调用此方法获取当前视频的帧率。"""
        return self.fps
    
    def _save_config(self):
        """私有方法：将内存中的配置写入 config.json 文件。"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            # 最好在实际项目中加上错误处理
            print(f"配置文件保存失败: {e}")
    def update_params(self, step_name: str, new_params: dict):
        """
        接收一个步骤的所有修改参数，合并到内存配置，并保存到文件。
        """
        if step_name not in self.config:
            print(f"警告: 配置中未找到步骤 {step_name}")
            return

        # 使用字典的 update() 方法，将 new_params 中的键值对合并到现有配置中
        self.config[step_name].update(new_params)
        
        # 立即保存到文件
        self._save_config()
        
    def update_config_param(self, step_name: str, key: str, value: Any):
        """
        用于 UI 模式切换，更新单个配置参数并立即保存。
        """
        if step_name not in self.config:
            return

        # 1. 更新内存中的配置
        self.config[step_name][key] = value

        # 2. 写入文件，确保修改生效
        self._save_config() 
                
    def update_step_status(self, step_name: str, status: str):
        self.step_statuses[step_name] = status
        print(f"[STATUS] {step_name}: {status}")
        
    def get_step_status(self, step_name: str) -> str:
        return self.step_statuses.get(step_name, "未运行")
    
    def get_varr_for_vis(self, step_name: str) -> Optional[xr.DataArray]:
        """
        根据步骤名称，从数据仓库中检索用于可视化的 xarray.DataArray (varr)。
        对于不需要视频数组作为背景的步骤，返回 None。
        """
        
        # 步骤名到数据仓库键的映射
        # 键名基于 Minian 的标准中间结果命名惯例
        data_key_map = {
            # 视频预处理步骤 (返回当前步骤的结果)
            'load_video_1': 'varr_glow',      # 步骤1：加载视频/去光晕后的结果
            'background_removal': 'varr_final_processed', # 步骤2：去除背景后的结果
            'denoise': 'varr_temporally_detrended',                # 步骤3：降噪后的结果
            'motion_correction': 'varr_mc',  # 步骤4：运动校正后的结果 (最干净的背景)
            
            # 种子点/初始化步骤 (以最干净的视频作为可视化背景)
            'seeds_init': 'video_for_seeds_vis',
            'ks_refine': 'video_for_seeds_vis',
            'merge_seeds': 'video_for_seeds_vis',
            'visualization_init': 'video_for_seeds_vis',
            
            # 非视频/图像可视化步骤 (返回 None，让 UI 根据 steps_results 处理)
            'peak_noise_ratio_refine': None,      # 曲线 (curve)
            'first_spatial_update_explore': None, # 探索 (exploration)
            'first_spatial_update_exec': None,    # CNMF更新 (cnmf_update)
            'first_temporal_update_explore': None,
            'first_temporal_update_exec': None,
            'save_data': None,                    # 无可视化 (none)
        }
        
        data_key = data_key_map.get(step_name)
        
        if data_key is None:
            # 对于不需要视频数组的步骤，直接返回 None
            return None
            
        # 使用已有的数据加载方法加载数据
        varr = self._load_data_from_repo(data_key)
        
        if varr is None:
            print(f"❌ 警告: 步骤 '{step_name}' 对应的可视化数据键 '{data_key}' 在数据仓库中找不到。")
            
        # 返回 xarray.DataArray
        return varr
    def run_load_video_1(self) -> bool:
        """
        步骤 1: 加载视频与去除光晕
        对应 Minian 的 load_videos 和 remove_glow 步骤。
        """
        step_name = 'load_video_1'
        self.update_step_status(step_name, "运行中")
        try:            
            params = self.get_step_params(step_name)
            
            # --- 修复开始: 处理 downsample 参数 ---
            ds_param = params.get('downsample', None)
            
            # 如果 downsample 是字符串 (例如 "dict(frame=1...)" 或 "{...}")，尝试转换
            if isinstance(ds_param, str):
                try:
                    # 尝试解析 Python 风格的 dict(...) 字符串
                    if ds_param.strip().startswith("dict("):
                        ds_param = eval(ds_param)
                    # 尝试解析 JSON 风格的字符串
                    else:
                        ds_param = json.loads(ds_param)
                except Exception as e:
                    self.log_output.append(f"⚠️ 警告: downsample 参数解析失败 ('{ds_param}')，将使用 None。错误: {e}")
                    ds_param = None
            # --- 修复结束 ---

            # 1. 从参数中提取 load_videos 需要的参数
            load_params = {
                'pattern': params.get('pattern', r"msCam[0-9]+\.avi$"),
                'dtype': params.get('dtype', 'uint16'),
                'downsample': ds_param, # 使用处理后的 ds_param
            }
            
            # 2. 调用 Minian 核心函数: load_videos
            self.log_output.append(f"-> 正在加载视频 (downsample={ds_param})...")
            varr = load_videos(vpath=self.video_folder, **load_params)
            
            # 确保缓存目录存在
            os.makedirs(self.data_path, exist_ok=True)
            
            # 调试保存 (可选)
            # varr.to_netcdf(os.path.join(self.data_path, "origin.nc"))

            # 3. 调用 Minian 核心函数: remove_glow
            self.log_output.append("-> 正在去除光晕...")
            varr_glow_removed = remove_glow(varr=varr)
            
            # 调试保存 (可选)
            # varr_glow_removed.to_netcdf(os.path.join(self.data_path, "varr_glow.nc"))
            
            # 4. 保存结果到数据仓库
            self._save_data_to_repo(varr_glow_removed, 'varr_glow')
            
            # 5. 更新 FPS
            if 'frame' in varr.coords and 'fs' in varr.coords['frame'].attrs:
                self.set_video_fps(varr.coords['frame'].attrs['fs'])
            
            self.log_output.append("✅ 视频加载与光晕去除完成。")
            self.update_step_status(step_name, "已完成")
            return True
            
        except Exception as e:
            import traceback
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            # 打印详细堆栈，方便调试
            print(traceback.format_exc()) 
            self.update_step_status(step_name, "错误")
            return False


    def run_denoise(self) -> bool:
        """
        步骤 2: 降噪 (时域带通滤波/空域滤波)
        根据参数中的 'method' 选择不同的降噪模式和参数。
        """
        step_name = 'denoise'
        # 修正输入键名：应与 load_video_1 的输出键一致
        input_key = 'varr_glow' 
        # 修正输出键名：用于下一步骤（如运动校正或去背景）
        output_key = 'varr_temporally_detrended' 
        
        self.update_step_status(step_name, "运行中")

        try:
            varr_in = self._load_data_from_repo(input_key)
            
            if varr_in is None:
                self.log_output.append(f"⚠️ 警告: 输入数据 ('{input_key}') 未找到，无法执行降噪。")
                self.update_step_status(step_name, "跳过/错误")
                return False
            if varr_in.dtype != np.float32:
                self.log_output.append(f"-> 正在将降噪输入数据类型从 {varr_in.dtype} 转换为 float32，以兼容 OpenCV。")
                # xarray.DataArray.astype() 会自动处理底层的 dask 数组
                varr_in = varr_in.astype(np.float32)
            # 1. 获取所有配置参数
            params = self.get_step_params(step_name)
            method = params.get('method', 'fft')
            
            # 打印一下当前拿到的所有参数，用于调试
            print(f"DEBUG: run_denoise 接收到的完整参数: {params}")

            call_kwargs = {}
            prefix_to_match = f"{method}_"
            
            for key, value in params.items():
                if key == 'method':
                    continue
                
                # 逻辑修正：严格匹配前缀
                if key.startswith(prefix_to_match):
                    # 剥离前缀： 'fft_low_cut' -> 'low_cut'
                    param_name = key[len(prefix_to_match):]
                    
                    # 类型安全转换
                    if ('ksize' in param_name or 'wnd' in param_name) and isinstance(value, list):
                        value = tuple(value)
                        
                    call_kwargs[param_name] = value
            
            # 打印最终传递给函数的参数
            print(f"DEBUG: 传递给 denoise 函数的参数: {call_kwargs}")
            # 4. 核心调用: denoise
            varr_out = denoise(
                varr_in,
                method=method,
                **call_kwargs
            )
            varr_out.to_netcdf(r"D:\Desktop\ZJU\SRTP\demo\minian_visual_cache\varr_temporally_detrended.nc")
            # 5. 保存结果
            self._save_data_to_repo(varr_out, output_key)
            
            self.log_output.append(f"✅ {method} 降噪完成。数据已保存到键 '{output_key}'。")
            self.update_step_status(step_name, "已完成")
            return True
            
        except NotImplementedError as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}。请检查参数 method 是否正确。")
            self.update_step_status(step_name, "错误")
            return False
        except Exception as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.update_step_status(step_name, "错误")
            return False

    def run_background_removal(self) -> bool:
        """
        步骤 3: 去除背景 (空域形态学滤波)
        """
        step_name = 'background_removal'
        input_key = 'varr_temporally_detrended'
        output_key = 'varr_final_processed' 
        self.update_step_status(step_name, "运行中")

        try:
            varr_in = self._load_data_from_repo(input_key)
            
            if varr_in is None:
                self.log_output.append(f"⚠️ 警告: 输入数据 ('{input_key}') 未找到，无法执行背景去除。")
                self.update_step_status(step_name, "跳过/错误")
                return False
            print(varr_in.shape)
            print(varr_in.dtype)
            params = self.get_step_params(step_name)
            method = params.get('method', 'tophat') # 默认为 tophat
            call_kwargs = {} # 用于传递给 remove_background 的参数字典
            
            print(f"DEBUG: run_background_removal 接收到的完整参数: {params}")

            self.log_output.append(f"-> 正在执行空域背景去除。通用模式='{method}'。")
            
            # 遍历参数，筛选并清理键名
            prefix_to_match = f"{method}_"
            
            for key, value in params.items():
                if key == 'method':
                    continue
                    
                # 检查参数键是否以当前 method 为前缀
                if key.startswith(prefix_to_match):
                    # 提取参数名：例如 'tophat_wnd' -> 'wnd'
                    param_name = key[len(prefix_to_match):]
                    
                    # 特殊类型处理：wnd 可能需要从列表转为元组
                    if ('wnd' in param_name or 'ksize' in param_name) and isinstance(value, list):
                        value = tuple(value)
                    
                    call_kwargs[param_name] = value
                    
            print(f"DEBUG: 传递给 remove_background 函数的参数: {call_kwargs}")
            
            # 核心调用: 使用您提供的 remove_background 函数
            varr_out = remove_background(
                varr_in,
                method=method,
                **call_kwargs # 使用解包的参数
            )
            varr_out.to_netcdf(r"D:\Desktop\ZJU\SRTP\demo\minian_visual_cache\varr_final_processed.nc")
            
            self._save_data_to_repo(varr_out, output_key)
            self.log_output.append("✅ 空域背景去除完成。")
            self.update_step_status(step_name, "已完成")
            return True
            
        except Exception as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.update_step_status(step_name, "错误")
            return False

    def run_motion_correction(self) -> bool:
        """
        步骤 4: 运动校正
        对应 Minian 的 estimate_motion 和 apply_shifts 步骤。
        """
        step_name = 'motion_correction'
        input_key = 'varr_final_processed' 
        output_key = 'varr_mc'
        self.update_step_status(step_name, "运行中")

        try:
            varr_in = self._load_data_from_repo(input_key)
            params = self.get_step_params(step_name)
            print(params)
            # 1. 估计运动
            self.log_output.append("-> 正在估计运动位移...")
            shifts = estimate_motion(varr_in, **params.get('estimate_motion_kwargs', {}))
            self._save_data_to_repo(shifts, 'shifts') # 保存位移数据
            
            # 2. 应用运动位移
            self.log_output.append("-> 正在应用运动校正...")
            varr_mc = apply_shifts(varr_in, shifts, **params.get('apply_shifts_kwargs', {}))
            
            self.steps_results[step_name] = (varr_in, varr_mc)
            
            # 输出varr_mc的类型
            print(type(varr_mc))
            # 输出varr_mc的shape
            print(varr_mc.shape)
            varr_mc.to_netcdf(r"D:\Desktop\ZJU\SRTP\demo\minian_visual_cache\mc.nc")

            # np.save(varr_mc,r"D:\Desktop\ZJU\SRTP\demo\minian_visual_cache\mc.npy")
            
            self._save_data_to_repo((varr_in,varr_mc), output_key)
            self.log_output.append("✅ 运动校正完成。")
            self.update_step_status(step_name, "已完成")
            return True

        except Exception as e:
            # 打印完整的错误追踪栈
            error_trace = traceback.format_exc()
            
            # 打印简短摘要
            self.log_output.append(f"❌ 运行【{step_name}】失败: {type(e).__name__} - {e}")
            # 打印详细追踪
            print(f"--- 详细追踪线 ---\n{error_trace}")
            self.log_output.append(f"--- 详细追踪栈 ---\n{error_trace}")
            
            # 修复 TypeError 崩溃：确保在失败时将结果设置为 None
            self.steps_results[step_name] = None 
            
            self.update_step_status(step_name, "错误")
            return False

    # 文件: minian_processor.py
    # 函数: run_seeds_init

    def run_seeds_init(self) -> bool:
        """
        步骤 5: 初始化种子 (Seeds Initialization)
        使用 seeds_init 函数计算初始空间组件的候选区域（种子）。
        """
        step_name = 'seeds_init'
        self.update_step_status(step_name, "运行中")
        try:
            # 🚨 移除对 os.environ["MINIAN_INTERMEDIATE"] 的设置和检查，由 UI 负责

            # 1. 加载运动校正后的视频 (修正导入语句)
            varr_mc_raw = self._load_data_from_repo('varr_mc') 
            
            if isinstance(varr_mc_raw, tuple) and len(varr_mc_raw) == 2:
                varr_in_for_seeds = varr_mc_raw[1]
            else:
                varr_in_for_seeds = varr_mc_raw
            
            if varr_in_for_seeds is None:
                self.log_output.append("⚠️ 警告: 运动校正后的视频数据 ('varr_mc') 未找到，无法执行种子初始化。")
                self.update_step_status(step_name, "跳过/错误")
                return False

            # 🚨 修复可视化：存储用于可视化背景的视频数组 (用于 UI 获取背景视频)
            # self._save_data_to_repo(varr_in_for_seeds, 'varr_seeds') 
            self._save_data_to_repo(varr_in_for_seeds, 'video_for_seeds_vis')
            
            params = self.get_step_params(step_name)
            print(f"DEBUG: run_seeds_init 接收到的参数: {params}")

            # 2. 确定方法
            method = params.get('method', 'rolling')
            
            # 3. 准备参数字典
            # 不要使用硬编码的 DEFAULT_PARAMS 来初始化，而是根据函数签名动态构建
            # 或者先定义默认值，然后用 params 覆盖它
            
            # 默认值定义 (仅作为兜底，若 config 中有值则会被覆盖)
            params_to_pass = {
                'wnd_size': 1000, 
                'stp_size': 500, 
                'max_wnd': 15, 
                'diff_thres': 3,
                'method': method
            }
            
            prefix_to_match = f"{method}_"
            
            for key, value in params.items():
                if key == 'method': continue
                
                # 如果参数带前缀 (如 rolling_wnd_size)，剥离前缀并覆盖默认值
                if key.startswith(prefix_to_match):
                    param_name = key[len(prefix_to_match):]
                    params_to_pass[param_name] = value
                
                # 如果参数本身就是无前缀的通用参数 (如某些配置可能直接存了 wnd_size)，也允许覆盖
                elif key in params_to_pass:
                    params_to_pass[key] = value

            # print(f"DEBUG: 最终传递给 seeds_init 的参数: {params_to_pass}")

            # 4. 调用函数
            seeds = seeds_init(varr_in_for_seeds, **params_to_pass)
                
            # 5. 检查 seeds_init 的返回结果
            if seeds is None or (hasattr(seeds, 'empty') and seeds.empty):
                self.log_output.append("❌ 核心函数 seeds_init 运行失败或未找到任何种子。请检查参数设置。")
                self.update_step_status(step_name, "错误")
                return False
                
            self.log_output.append(
                f"-> 种子初始化完成，共找到 {len(seeds)} 个候选区域。"
            )

            # 计算最大投影作为种子可视化背景
            self.log_output.append("-> 正在计算种子可视化的最大投影图...")
            max_proj = varr_in_for_seeds.max(dim='frame').compute()
            
            # 保存种子数据到'varr_seeds' (pandas.DataFrame)
            self._save_data_to_repo(seeds, 'varr_seeds')
            
            # 保存最大投影图到'max_proj_seeds' (xarray.DataArray)
            self._save_data_to_repo(max_proj, 'max_proj_seeds')
            
            self.log_output.append("✅ 初始种子 (seeds) 已保存。")
            self.update_step_status(step_name, "已完成")
            return True
            
        except Exception as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.log_output.append(f"--- 详细追踪栈 ---\n{traceback.format_exc()}")
            self.update_step_status(step_name, "错误")
            return False

    def run_noise_freq_exploration(self) -> tuple:
        """
        噪声频率探索与PNR曲线计算 (已修正版本)
        
        - 确保调用 smooth_sig 时传入了 'fs'。
        - (已修正) 确保从 `varr_seeds` (上一步的种子) 中采样，而不是生成随机坐标。
        """

        step_name = 'noise_freq_exploration'
        input_key = 'varr_temporally_detrended'
        # 🔴 修正 1: 声明种子的输入键
        seeds_input_key = 'varr_seeds' 

        try:
            # ==== Step 1: 参数与数据准备 ====
            self.update_step_status(step_name, "运行中")
            print(f"开始运行")
            varr_in = self._load_data_from_repo(input_key) # (frame, height, width)
            varr_in = varr_in.chunk(dict(frame=-1))
            
            if varr_in is None:
                raise ValueError("无法加载输入数据 (varr_in)。")
                
            # 🔴 修正 2: 加载 `seeds_init` 步骤生成的真实种子
            seeds_all_xr = self._load_data_from_repo(seeds_input_key)
            if seeds_all_xr is None:
                raise ValueError(f"无法加载种子数据 ('{seeds_input_key}')。请先运行 'seeds_init' 步骤。")

            # 🔴 修正 3: 将 Xarray 种子转换回 DataFrame 以便采样
            # (基于 initialization.py 中 seeds_init 的返回结构)
            try:
                seeds_df = pd.DataFrame({
                    'height': seeds_all_xr.coords['height'].values,
                    'width': seeds_all_xr.coords['width'].values
                })
            except Exception as e:
                print(f"转换种子数据失败: {e}. seeds_all_xr 结构: {seeds_all_xr}")
                raise

            n_frames, height, width = varr_in.shape
            print(f"数据准备完成 (f, h, w): {varr_in.shape}")
            
            # 参数读取
            params = self.get_step_params(step_name)
            noise_freq_candidates = np.array(params.get('noise_freq_list'))
            self.log_output.append(f"-> 探索噪声频率: {noise_freq_candidates} Hz")
            print(f"参数情况: {params}")
            print(f"探索噪声频率: {noise_freq_candidates} Hz")
            fs = float(params.get('fs', 30.0))
            
            # 🔴 修正 4: 从真实的种子 DataFrame 中采样
            n_samples_req = int(params.get('n_samples', 6)) # 您要求的 6
            
            if len(seeds_df) > n_samples_req:
                # 从真实种子中随机抽取 n_samples_req 个
                sample_seeds = seeds_df.sample(n=n_samples_req, random_state=42)
            else:
                # 如果总种子数少于要求，则使用所有种子
                sample_seeds = seeds_df
            
            # 更新 n_samples 为实际采样的数量
            n_samples = len(sample_seeds)
            print(f"已从 {len(seeds_df)} 个总种子中成功采样 {n_samples} 个。")
            

            print(f"开始计算")
            # ==== Step 2: 最佳频率探索 (基于所有像素点) ====
            best_pnr_mean_all = -np.inf
            best_noise_freq = None
            all_freq_pnr_means = {}
            print(f"开始循环")
            
            for freq in noise_freq_candidates:
                # 1. 分离信号与噪声
                print(f"进入循环，频率: {freq} Hz")
                signal_all = smooth_sig(varr_in, freq, fs, method="butter", btype="low")
                print(f"signal_all.shape {signal_all.shape}")
                noise_all = smooth_sig(varr_in, freq, fs, method="butter", btype="high")
                print(f"noise_all.shape {noise_all.shape}")
                
                 # 计算真正的信号幅度 (Amplitude)
                signal_baseline = signal_all.min('frame') 
                signal_amplitude = signal_all.max('frame') - signal_baseline
                
                # PNR = 幅度 / 噪声标准差
                pnr_all = signal_amplitude / noise_all.std('frame')
                
                # 3. 计算 PNR 均值
                # print(f"DEBUG: 开始计算频率 {freq} 的 PNR 均值...")
                pnr_mean_current_float = pnr_all.mean().compute().item()
                # print(f"DEBUG: 频率 {freq} 的 PNR 均值计算完成: {pnr_mean_current_float:.4f}")
                all_freq_pnr_means[freq] = pnr_mean_current_float

                print(f"For frequency {freq}, the PNR mean is {pnr_mean_current_float:.4f}.Current best is {best_pnr_mean_all:.4f}.")
                self.log_output.append(f"For frequency {freq}, the PNR mean is {pnr_mean_current_float:.4f}.Current best is {best_pnr_mean_all:.4f}.")
                
                
                # 4. 筛选最佳频率
                if pnr_mean_current_float > best_pnr_mean_all:
                    best_pnr_mean_all = pnr_mean_current_float
                    best_noise_freq = freq

            if best_noise_freq is None:
                raise RuntimeError("未能确定最佳噪声频率。")

            # self.log_output.append(f"最佳噪声截止频率确定为: {best_noise_freq} Hz (所有像素点PNR均值: {best_pnr_mean_all:.4f})")
            print(f"最佳噪声截止频率确定为: {best_noise_freq} Hz (所有像素点PNR均值: {best_pnr_mean_all:.4f})")
            self.save_best_freq_to_params(best_noise_freq) 
            print(f"最佳噪声频率已保存")
            
            # ==== Step 3: 基于最佳频率计算种子点数据 (用于返回) ====
            sample_coords_xr = sample_seeds.to_xarray().rename({'index': 'sample'})

            # b. 使用 .sel 和 DataArray 进行点对点索引
            #    这将返回一个 (frame, sample) 的 DataArray
            varr_samples_sel = varr_in.sel(
                height=sample_coords_xr["height"], 
                width=sample_coords_xr["width"]
            )
            
            # c. 转换为 (sample, frame) 以便 smooth_sig 处理
            varr_samples_chk = varr_samples_sel.transpose('sample', 'frame')
            
            print(f"种子点数据提取完成，形状: {varr_samples_chk.shape}") # (应为 6, 600)
            
            # 1. 再次分离信号与噪声 (仅针对种子点)
            signal_best = smooth_sig(varr_samples_chk, best_noise_freq, fs, method="butter", btype="low").compute()
            noise_best = smooth_sig(varr_samples_chk, best_noise_freq, fs, method="butter", btype="high").compute()

            # 2. 计算种子点的 PNR
            if not isinstance(signal_best, xr.DataArray):
                 signal_best = xr.DataArray(signal_best, dims=['sample', 'frame'])
                 noise_best = xr.DataArray(noise_best, dims=['sample', 'frame'])

            signal_best_baseline = signal_best.mean('frame')
            pnr_best = (signal_best.max('frame') - signal_best_baseline) / noise_best.std('frame')
            
            # ==== Step 4: 转换结构，并确保返回格式不变 ====
            
            # 1. 计算信号的基线 (每个样本的均值)
            # signal_best 形状是 (samples, frames)
            baselines = signal_best.min(dim='frame') 
            
            # 2. 将噪声数据加上这个基线，使其在视觉上与信号重合
            # 注意：利用广播机制 (samples, frames) + (samples,)
            noise_best_visual = noise_best + baselines
            
            # 3. 导出数据
            signals_arr = np.expand_dims(signal_best.values, axis=0)
            # 使用调整过基线的噪声数据用于显示
            noises_arr = np.expand_dims(noise_best_visual.values, axis=0) 
            pnrs_values = np.expand_dims(pnr_best.values, axis=0)
            
            pnrs_mean = np.array([np.mean(pnr_best.values)]) 
            
            noise_freq_list = np.array([best_noise_freq])
            print(f"数据转换完成")
            
            # ==== Step 5: 返回结果 ====
            self.update_step_status(step_name, "完成 ✅")
            return (
                "成功",
                sample_seeds, # sample_seeds 现在是包含 (height, width) 的 DataFrame
                pnrs_mean,
                pnrs_values,
                signals_arr,
                noises_arr,
                noise_freq_list,
                fs
            )

        except Exception as e:
            # 打印完整的错误堆栈以便调试
            import traceback
            traceback.print_exc() 
            
            self.update_step_status(step_name, "失败 ❌")
            self.log_output.append(f"[ERROR] 噪声频率探索失败: {str(e)}")
            print(f"[ERROR] 噪声频率探索失败: {str(e)}")
            return ("失败", None, None, None, None, None, None, None)

    # ... (save_best_freq_to_params 保持不变) ...
    # 占位函数（您要求我先那一个函数填充，这个后面再给出）
    def save_best_freq_to_params(self, freq: float):
        """
        将最佳频率保存到后续步骤的参数中。
        """
        # 实际实现中，这里会调用类似 self.set_param('next_step_name', 'noise_freq', freq) 的方法
        print(f"[INFO] 最佳频率 {freq} Hz 已保存供后续步骤使用。")
        # 辅助函数 (从 visualization.py 复制过来，确保 processor 内部可用)
        
    def _convert_seeds_to_df(self,seeds_data: Union[pd.DataFrame, xr.DataArray]) -> pd.DataFrame:
        """辅助函数：将 XArray 种子转换为规范的 DataFrame。"""
        if isinstance(seeds_data, xr.DataArray):
            try:
                seeds_df = pd.DataFrame({
                    'height': seeds_data.coords['height'].values,
                    'width': seeds_data.coords['width'].values
                })
            except KeyError:
                seeds_df = seeds_data.rename("seeds").to_dataframe().reset_index()
                if 'height' not in seeds_df.columns or 'width' not in seeds_df.columns:
                    if 'dim_0' in seeds_df.columns and 'dim_1' in seeds_df.columns:
                        seeds_df['height'] = seeds_df['dim_0']
                        seeds_df['width'] = seeds_df['dim_1']
                    else:
                        raise ValueError("输入DataArray必须包含height/width或dim_0/dim_1维度")
            return seeds_df
        else:
            return seeds_data.copy()

    def run_peak_noise_ratio_refine(self) -> bool:
        """
        ...
        """
        step_name = 'peak_noise_ratio_refine'
        input_key = 'varr_seeds' 
        
        output_key_kept = 'seeds_pnr_kept'
        output_key_removed = 'seeds_pnr_removed'
        
        self.update_step_status(step_name, "运行中")

        try:
            varr_mc_tuple = self._load_data_from_repo('varr_mc')
            if isinstance(varr_mc_tuple, tuple) and len(varr_mc_tuple) == 2:
                varr_mc = varr_mc_tuple[1]
            else:
                varr_mc = varr_mc_tuple
                
            seeds_init_xr = self._load_data_from_repo(input_key)
            params = self.get_step_params(step_name)
            
            # 🔴 修正 1: 获取采样频率 (fs)
            fs = self.get_video_fps()
            
            # 严格参数验证
            valid_params = {}
            for k, v in params.items():
                clean_key = k
                # 🔴 修正 2: 允许 'fs' (但 'fs' 不应来自UI)
                if clean_key in ['noise_freq', 'thres']:  
                    valid_params[clean_key] = v
            
            # 🔴 修正 3: 将 processor 的 'fs' 强行注入参数
            valid_params['fs'] = fs
            
            self.log_output.append(f"-> 原始参数: {params}")
            self.log_output.append(f"-> 有效参数 (含fs): {valid_params}")
            
            if not valid_params.get('noise_freq'):
                self.log_output.append("⚠️ 警告: noise_freq参数缺失，使用默认值0.25")
                valid_params['noise_freq'] = 0.25
                
            # 准备数据格式
            self.log_output.append("-> 正在准备数据格式...")
            
            # 处理varr_mc (xarray对象)
            if hasattr(varr_mc, 'chunk'):
                varr_mc = varr_mc.chunk({'height': -1, 'width': -1, 'frame': -1}).persist()
            
            # 🔴 修正: 确保 seeds_init 是 pandas.DataFrame (使用 visualization.py 中的辅助函数逻辑)
            self.log_output.append(f"-> seeds_init类型: {type(seeds_init_xr)}")
            try:
                seeds_init_df = self._convert_seeds_to_df(seeds_init_xr)
            except Exception as e:
                self.log_output.append(f"❌ 无法转换 'varr_seeds' 为 DataFrame: {e}")
                raise
            
            self.log_output.append("-> 正在计算 PNR 并精修种子点...")
            try:
                # pnr_refine (from initialization.py) 返回一个 DataFrame
                # 包含一个布尔列 "mask_pnr"
                seeds_with_mask, pnrs, gmm = pnr_refine(
                    varr_mc,
                    seeds_init_df, # 传入 DataFrame
                    **valid_params
                )
            except Exception as e:
                self.log_output.append(f"❌ PNR计算失败: {str(e)}")
                print(f"❌ PNR计算失败: {str(e)}")
                raise
            
            # 🔴 修正: 将种子分为 "保留" 和 "移除"
            if 'mask_pnr' not in seeds_with_mask.columns:
                self.log_output.append(f"❌ 错误: 'pnr_refine' 未返回 'mask_pnr' 列。")
                raise ValueError("pnr_refine的输出缺少'mask_pnr'列")

            seeds_kept = seeds_with_mask[seeds_with_mask['mask_pnr'] == True]
            seeds_removed = seeds_with_mask[seeds_with_mask['mask_pnr'] == False]
            
            # 🔴 修正: 保存两组分离的种子
            self._save_data_to_repo(seeds_kept, output_key_kept)
            self._save_data_to_repo(seeds_removed, output_key_removed)
            
            # (可选) 保存 PNR 值和 GMM 模型
            self._save_data_to_repo(pnrs, 'pnrs')
            self._save_data_to_repo(gmm, 'gmm_model') 
            
            self.log_output.append(f"✅ PNR 精修完成。")
            self.log_output.append(f"-> {len(seeds_kept)} 个种子被保留 (白色)。")
            self.log_output.append(f"-> {len(seeds_removed)} 个种子被移除 (红色)。")
            print(f"✅ PNR 精修完成。保留: {len(seeds_kept)}, 移除: {len(seeds_removed)}")
            
            # 🔴 修正: 移除旧的 matplotlib 可视化代码
            # (从 "生成可视化结果..." 到 "self.log_output.append(f"可视化生成失败: {str(e)}")" 
            #  的所有代码块都应被删除, 因为现在由 main_pipeline_window.py 处理)
            
            self.update_step_status(step_name, "已完成")
            
            # 🔴 修正: 返回简单的 True
            return True

        except Exception as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.log_output.append(f"--- 详细追踪栈 ---\n{traceback.format_exc()}")
            print(f"--- 详细追踪栈 ---\n{traceback.format_exc()}")
            print(traceback.format_exc())
            self.update_step_status(step_name, "错误")
            return False

    def run_ks_refine(self) -> bool:
        """
        步骤 7: KS 检验精修
        (已修改为在 'pnr_refine' 步骤的结果上进行筛选，并保存 'kept' 和 'removed' 种子)
        """
        step_name = 'ks_refine'
        
        # 🔴 修正 1: 输入键必须是 PNR 步骤保留的种子
        input_key = 'seeds_pnr_kept' 
        
        # 🔴 修正 2: 定义新的、唯一的输出键
        output_key_kept = 'seeds_ks_kept'
        output_key_removed = 'seeds_ks_removed'
        
        self.update_step_status(step_name, "运行中")

        try:
            # 1. 加载运动校正视频 (用于 ks_refine)
            varr_mc_tuple = self._load_data_from_repo('varr_mc')
            if isinstance(varr_mc_tuple, tuple) and len(varr_mc_tuple) == 2:
                varr_mc = varr_mc_tuple[1]
            else:
                varr_mc = varr_mc_tuple

            if varr_mc is None:
                 raise ValueError("无法加载 'varr_mc' 数据。")
            
            self.log_output.append("-> 正在重分块 (Rechunking) 'frame' 维度...")
            varr_mc = varr_mc.chunk({'frame': -1})
            
            # 2. 加载 PNR 步骤保留的种子 (DataFrame)
            seeds_to_test = self._load_data_from_repo(input_key)
            if seeds_to_test is None:
                raise ValueError(f"无法加载上一步的种子数据 ('{input_key}')。请先运行 'peak_noise_ratio_refine' 步骤。")
            if not isinstance(seeds_to_test, pd.DataFrame):
                self.log_output.append(f"-> 警告: 输入的种子不是 DataFrame (类型: {type(seeds_to_test)})，尝试转换...")
                seeds_to_test = self._convert_seeds_to_df(seeds_to_test)

            # 3. 获取参数 (ks_refine 只接受 'sig')
            params = self.get_step_params(step_name)
            valid_params = {}
            if 'sig' in params:
                valid_params['sig'] = float(params['sig'])
            
            self.log_output.append(f"-> 正在对 {len(seeds_to_test)} 个种子使用 KS 检验精修...")
            self.log_output.append(f"-> 有效参数: {valid_params}")

            # 4. 调用核心函数
            # ks_refine 返回一个 *新的* DataFrame，其中添加了 'mask_ks' 列
            seeds_with_mask = ks_refine(varr_mc, seeds_to_test, **valid_params)
            
            # 5. 🔴 修正 3: 根据 'mask_ks' 分离种子
            if 'mask_ks' not in seeds_with_mask.columns:
                self.log_output.append(f"❌ 错误: 'ks_refine' 未返回 'mask_ks' 列。")
                raise ValueError("ks_refine 的输出缺少 'mask_ks' 列")

            # 注意：ks_refine 的 'mask_ks' 是 True 表示 *通过* (p-value < sig)
            seeds_kept = seeds_with_mask[seeds_with_mask['mask_ks'] == True]
            seeds_removed = seeds_with_mask[seeds_with_mask['mask_ks'] == False]
            
            # 6. 保存两组分离的种子
            self._save_data_to_repo(seeds_kept, output_key_kept)
            self._save_data_to_repo(seeds_removed, output_key_removed)

            self.log_output.append(f"✅ KS 检验与过滤完成。")
            self.log_output.append(f"-> {len(seeds_kept)} 个种子被保留 (白色)。")
            self.log_output.append(f"-> {len(seeds_removed)} 个种子被移除 (红色)。")
            print(f"✅ KS 检验完成。保留: {len(seeds_kept)}, 移除: {len(seeds_removed)}")

            self.update_step_status(step_name, "已完成")
            return True

        except Exception as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.log_output.append(f"--- 详细追踪栈 ---\n{traceback.format_exc()}")
            print(f"--- 详细追踪栈 ---\n{traceback.format_exc()}")
            print(traceback.format_exc())
            self.update_step_status(step_name, "错误")
            return False

    def run_merge_seeds(self) -> bool:
        """
        步骤 8: 合并种子点
        (已修改为在 'ks_refine' 步骤的结果上进行筛选，并保存 'kept' 和 'removed' 种子)
        """
        step_name = 'merge_seeds'
        
        # 🔴 修正 1: 输入键
        input_key = 'seeds_ks_kept'
        
        # 🔴 修正 2: 唯一的输出键
        output_key_kept = 'seeds_merged_kept'
        output_key_removed = 'seeds_merged_removed'
        
        self.update_step_status(step_name, "运行中")

        try:
            # 1. 加载视频
            varr_mc_tuple = self._load_data_from_repo('varr_mc')
            if isinstance(varr_mc_tuple, tuple) and len(varr_mc_tuple) == 2:
                varr_mc = varr_mc_tuple[1]
            else:
                varr_mc = varr_mc_tuple

            # 🔴 修正 3: Rechunk 'frame' 维度
            # (seeds_merge -> adj_corr -> smooth_corr -> filt_fft 
            #  都需要完整的 'frame' 维度)
            if varr_mc is None:
                 raise ValueError("无法加载 'varr_mc' 数据。")
            self.log_output.append("-> 正在重分块 (Rechunking) 'frame' 维度...")
            varr_mc = varr_mc.chunk({'frame': -1})

            # 2. 加载 'ks_refine' 保留的种子
            seeds_to_merge = self._load_data_from_repo(input_key)
            if seeds_to_merge is None:
                raise ValueError(f"无法加载上一步的种子数据 ('{input_key}')。请先运行 'ks_refine' 步骤。")
            if not isinstance(seeds_to_merge, pd.DataFrame):
                self.log_output.append(f"-> 警告: 输入的种子不是 DataFrame (类型: {type(seeds_to_merge)})，尝试转换...")
                seeds_to_merge = self._convert_seeds_to_df(seeds_to_merge) # 使用辅助函数

            params = self.get_step_params(step_name)
            
            # 3. 计算 max_proj (这个 'max_proj' 与 'max_proj_seeds' 不同，
            #    它是在 mc 视频上计算的，用于 merge)
            self.log_output.append("-> 正在计算 Max Projection (用于合并)...")
            max_proj = varr_mc.max('frame').compute()
            # (注意: 'max_proj_seeds' 仍用于可视化背景)
            
            # 4. 合并种子点
            self.log_output.append(f"-> 正在对 {len(seeds_to_merge)} 个种子进行合并...")
            
            # seeds_merge 返回一个 DataFrame，其中添加了 'mask_mrg' 列
            seeds_with_mask = seeds_merge(varr_mc, max_proj, seeds_to_merge, **params)
            
            # 🔴 修正 4: 根据 'mask_mrg' 分离种子
            if 'mask_mrg' not in seeds_with_mask.columns:
                self.log_output.append(f"❌ 错误: 'seeds_merge' 未返回 'mask_mrg' 列。")
                raise ValueError("seeds_merge 的输出缺少 'mask_mrg' 列")

            # 'mask_mrg' 为 True 表示*保留*
            seeds_kept = seeds_with_mask[seeds_with_mask['mask_mrg'] == True]
            seeds_removed = seeds_with_mask[seeds_with_mask['mask_mrg'] == False]

            # 5. 保存两组分离的种子
            self._save_data_to_repo(seeds_kept, output_key_kept)
            self._save_data_to_repo(seeds_removed, output_key_removed)

            self.log_output.append(f"✅ 种子点合并完成。")
            self.log_output.append(f"-> {len(seeds_kept)} 个种子被保留 (白色)。")
            self.log_output.append(f"-> {len(seeds_removed)} 个种子被移除 (红色)。")
            print(f"✅ 种子点合并完成。保留: {len(seeds_kept)}, 移除: {len(seeds_removed)}")

            self.update_step_status(step_name, "已完成")
            return True

        except Exception as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.log_output.append(f"--- 详细追踪栈 ---\n{traceback.format_exc()}")
            print(f"--- 详细追踪栈 ---\n{traceback.format_exc()}")
            print(traceback.format_exc())
            self.update_step_status(step_name, "错误")
            return False

    def run_visualization_init(self) -> bool:
        """
        步骤 9: 初始化 (A, C, b, f)
        (已修正为使用分批处理来避免 ArrayMemoryError)
        """
        step_name = 'visualization_init'
        self.update_step_status(step_name, "运行中")

        try:            
            # 0. 加载数据
            varr_mc_tuple = self._load_data_from_repo('varr_mc')
            if isinstance(varr_mc_tuple, tuple) and len(varr_mc_tuple) == 2:
                varr_mc = varr_mc_tuple[1]
            else:
                varr_mc = varr_mc_tuple
            
            seeds_to_init = self._load_data_from_repo('seeds_merged_kept')
            if seeds_to_init is None:
                raise ValueError("未找到 'seeds_merged_kept' 数据。请先运行 'merge_seeds' 步骤。")
            if not isinstance(seeds_to_init, pd.DataFrame):
                 seeds_to_init = self._convert_seeds_to_df(seeds_to_init)
            
            if varr_mc is None:
                 raise ValueError("无法加载 'varr_mc' 数据。")
            
            # 1. (关键) 重分块并用 0 填充
            self.log_output.append("-> 正在重分块 (Rechunking) 'frame' 维度...")
            varr_mc = varr_mc.chunk({'frame': -1})
            self.log_output.append("-> 正在用 0 填充 (Fills NaNs with 0)...")
            varr_mc = varr_mc.fillna(0).astype(np.float32).persist()

            params = self.get_step_params(step_name)
            intpath = os.environ.get("MINIAN_INTERMEDIATE", "./intermediate_data")

            # 2. 🔴 内存错误修复：分批处理 🔴
            
            batch_size = 500  # 一次处理 500 个种子
            n_seeds = len(seeds_to_init)
            A_init_list = [] # 存储每个批次的 A_init

            self.log_output.append(f"-> 正在对 {n_seeds} 个种子分批 (每批 {batch_size} 个) 初始化空间足迹 A...")

            for i in range(0, n_seeds, batch_size):
                batch_start = i
                batch_end = min(i + batch_size, n_seeds)
                self.log_output.append(f"--> 正在处理批次 {batch_start} to {batch_end}...")
                
                seeds_batch_df = seeds_to_init.iloc[batch_start:batch_end]
                
                # (A) 空间初始化 A (针对批次)
                A_batch = initA(varr_mc, seeds_batch_df, **params.get('initA_kwargs', {}))
                
                # (B) 强制计算这个小批次
                # 这将只加载计算这 500 个种子所需的数据
                # 应该可以避免 551 MiB / 826 MiB 的错误
                A_batch_computed = A_batch.persist() 
                
                A_init_list.append(A_batch_computed)
                self.log_output.append(f"--> 批次 {batch_start} 完成.")

            # (C) 合并所有已计算的批次
            self.log_output.append("-> 所有批次 A 初始化完成，正在合并...")
            A_init = xr.concat(A_init_list, dim="unit_id")
            
            # 3. 时间初始化 C (依赖于 A_init)
            self.log_output.append("-> 正在初始化时间序列 C...")
            A_init = A_init.persist() # 确保合并后的 A 在内存中
            C_init = initC(varr_mc, A_init) 
            
            # 4. A 和 C 的初始合并
            self.log_output.append("-> 正在执行 A 和 C 的初始合并...")
            A_mrg, C_mrg = unit_merge(A_init, C_init, **params.get('init_merge_kwargs', {}))
            
            # 5. 计算初始背景 b 和 f
            self.log_output.append("-> 正在计算初始背景 b 和 f...")
            A_mrg = A_mrg.persist()
            C_mrg = C_mrg.persist()
            b_init, f_init = update_background(varr_mc, A_mrg, C_mrg)

            # 6. 保存最终的 A, C, b, f
            self.log_output.append("-> 正在保存 A, C, b, f...")
            
            A = save_minian(A_mrg.rename("A_init"), dpath=intpath, overwrite=True)
            C = save_minian(
                C_mrg.rename("C_init"), 
                dpath=intpath, 
                overwrite=True, 
                chunks=params.get('C_chunks', {"unit_id": 1, "frame": -1})
            )
            b = save_minian(b_init.rename("b_init"), dpath=intpath, overwrite=True)
            f = save_minian(f_init.rename("f_init"), dpath=intpath, overwrite=True)

            self._save_data_to_repo(A, "A_init")
            self._save_data_to_repo(C, "C_init")
            self._save_data_to_repo(b, "b_init")
            self._save_data_to_repo(f, "f_init")
            
            self.log_output.append("✅ A, C, b, f 初始化完成。")
            self.update_step_status(step_name, "已完成")
            return True
            
        except Exception as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.log_output.append(f"--- 详细追踪栈 ---\n{traceback.format_exc()}")
            print(f"--- 详细追踪栈 ---\n{traceback.format_exc()}")
            print(traceback.format_exc())
            self.update_step_status(step_name, "错误")
            return False

    def get_exploration_result(self, step_name: str):
        return self.exploration_results.get(step_name)

    def get_exploration_penalties(self, step_name: str):
        data = self.exploration_results.get(step_name, {})
        return data.get("penalty_list", [])

    def set_exploration_state(self, step_name: str, state: dict):
        self.exploration_state[step_name] = state

    def get_exploration_state(self, step_name: str):
        return self.exploration_state.get(step_name, {})

    def _normalize_size_thres(self, val):
        """Normalize UI/config value for size_thres into either None or a (low, high) tuple.

        Handles string inputs such as 'null'/'None' and JSON lists.
        """
        # direct None
        if val is None:
            return None

        # strings: try to interpret
        if isinstance(val, str):
            s = val.strip().lower()
            if s in ("null", "none", "nan", ""):
                return None
            # try JSON-like list/tuple
            try:
                parsed = json.loads(val)
                val = parsed
            except Exception:
                # leave as-is (fall through)
                pass

        # list/tuple -> ensure length 2 and convert inner 'null' to None
        if isinstance(val, (list, tuple)):
            out = []
            for x in list(val)[:2]:
                if x is None:
                    out.append(None)
                elif isinstance(x, str) and str(x).strip().lower() in ("null", "none", "nan", ""):
                    out.append(None)
                else:
                    try:
                        out.append(float(x))
                    except Exception:
                        out.append(None)
            # pad to length 2
            if len(out) == 1:
                out.append(None)
            return (out[0], out[1])

        # single numeric -> treat as lower bound
        if isinstance(val, (int, float)):
            try:
                return (float(val), None)
            except Exception:
                return None

        # fallback
        return None

    def _diagnose_da(self, da: xr.DataArray, label: str, per_unit: bool = False) -> Dict[str, Any]:
        """Collect numeric diagnostics for a DataArray and append concise logs.

        This is used to trace why downstream filtering drops all units.
        """
        try:
            if da is None:
                diag = {"label": label, "available": False, "reason": "None"}
                self.log_output.append(f"[DIAG] {label}: None")
                return diag

            dims = list(getattr(da, "dims", []))
            sizes = {d: int(da.sizes.get(d, 0)) for d in dims}
            n_elem = int(np.prod([max(1, sizes[d]) for d in dims])) if dims else int(da.size)

            finite_cnt = int(xr.where(np.isfinite(da), 1, 0).sum().compute().values)
            pos_cnt = int((da > 0).sum().compute().values)
            zero_cnt = int(xr.where(da == 0, 1, 0).sum().compute().values)
            neg_cnt = int((da < 0).sum().compute().values)

            try:
                g_min = float(da.min().compute().values)
            except Exception:
                g_min = float("nan")
            try:
                g_max = float(da.max().compute().values)
            except Exception:
                g_max = float("nan")
            try:
                g_mean = float(da.mean().compute().values)
            except Exception:
                g_mean = float("nan")
            try:
                abs_sum = float(np.abs(da).sum().compute().values)
            except Exception:
                abs_sum = float("nan")

            diag: Dict[str, Any] = {
                "label": label,
                "available": True,
                "dims": dims,
                "sizes": sizes,
                "n_elem": n_elem,
                "finite_cnt": finite_cnt,
                "non_finite_cnt": int(max(0, n_elem - finite_cnt)),
                "pos_cnt": pos_cnt,
                "zero_cnt": zero_cnt,
                "neg_cnt": neg_cnt,
                "pos_ratio": float(pos_cnt / n_elem) if n_elem > 0 else 0.0,
                "zero_ratio": float(zero_cnt / n_elem) if n_elem > 0 else 0.0,
                "neg_ratio": float(neg_cnt / n_elem) if n_elem > 0 else 0.0,
                "global_min": g_min,
                "global_max": g_max,
                "global_mean": g_mean,
                "abs_sum": abs_sum,
            }

            if per_unit and "unit_id" in dims and da.sizes.get("unit_id", 0) > 0:
                red_dims = [d for d in dims if d != "unit_id"]
                if len(red_dims) > 0:
                    pos_per_unit = (da > 0).sum(red_dims).compute().values
                    abs_per_unit = np.abs(da).sum(red_dims).compute().values
                    pos_per_unit = np.asarray(pos_per_unit, dtype=np.float64)
                    abs_per_unit = np.asarray(abs_per_unit, dtype=np.float64)

                    diag["unit_pos_min"] = float(np.nanmin(pos_per_unit)) if pos_per_unit.size else 0.0
                    diag["unit_pos_median"] = float(np.nanmedian(pos_per_unit)) if pos_per_unit.size else 0.0
                    diag["unit_pos_max"] = float(np.nanmax(pos_per_unit)) if pos_per_unit.size else 0.0
                    diag["unit_abs_min"] = float(np.nanmin(abs_per_unit)) if abs_per_unit.size else 0.0
                    diag["unit_abs_median"] = float(np.nanmedian(abs_per_unit)) if abs_per_unit.size else 0.0
                    diag["unit_abs_max"] = float(np.nanmax(abs_per_unit)) if abs_per_unit.size else 0.0
                    diag["units_pos_gt0"] = int((pos_per_unit > 0).sum())
                    diag["units_abs_gt0"] = int((abs_per_unit > 1e-12).sum())

            self.log_output.append(
                f"[DIAG] {label}: shape={tuple(da.shape)}, pos/zero/neg=({pos_cnt}/{zero_cnt}/{neg_cnt}), "
                f"min/mean/max=({g_min:.3e}/{g_mean:.3e}/{g_max:.3e}), abs_sum={abs_sum:.3e}"
            )
            if "units_pos_gt0" in diag:
                self.log_output.append(
                    f"[DIAG] {label}: units_pos_gt0={diag['units_pos_gt0']}, units_abs_gt0={diag['units_abs_gt0']}, "
                    f"unit_pos(min/med/max)=({diag['unit_pos_min']:.2f}/{diag['unit_pos_median']:.2f}/{diag['unit_pos_max']:.2f})"
                )
            return diag
        except Exception as e:
            self.log_output.append(f"[DIAG] {label} 诊断失败: {e}")
            return {"label": label, "available": False, "reason": str(e)}

    def run_first_spatial_update_explore(self) -> dict:
        step_name = "first_spatial_update_explore"
        self.update_step_status(step_name, "运行中")
        try:
            varr_mc_raw = self._load_data_from_repo("varr_mc")
            varr_mc = varr_mc_raw[1] if isinstance(varr_mc_raw, tuple) else varr_mc_raw
            # print("DEBUG:MC data loaded successfully.")

            A_init = self._load_data_from_repo("A_init")
            C_init = self._load_data_from_repo("C_init")
            sn_spatial = self._load_data_from_repo("sn_spatial")
            b_init = self._load_data_from_repo("b_init")
            f_init = self._load_data_from_repo("f_init")
            
            # print("DEBUG:Initialized data loaded successfully.")

            if varr_mc is None or A_init is None or C_init is None:
                raise ValueError("缺少 first_spatial_update_explore 所需输入数据(varr_mc/A_init/C_init)")

            # 1) 输入一致性修复：A/C 对齐 unit_id
            if "unit_id" in A_init.coords and "unit_id" in C_init.coords:
                common_units = np.intersect1d(
                    A_init.coords["unit_id"].values,
                    C_init.coords["unit_id"].values,
                )
                if len(common_units) == 0:
                    raise ValueError("A_init 与 C_init 没有公共 unit_id")
                if len(common_units) != len(A_init.coords["unit_id"]) or len(common_units) != len(C_init.coords["unit_id"]):
                    self.log_output.append(
                        f"⚠️ A_init/C_init unit_id 不一致，已自动对齐到 {len(common_units)} 个公共单元。"
                    )
                A_init = A_init.sel(unit_id=common_units)
                C_init = C_init.sel(unit_id=common_units)
                self._save_data_to_repo(A_init, "A_init")
                self._save_data_to_repo(C_init, "C_init")

            # 2) 自动补齐/修复 sn_spatial
            need_recompute_sn = sn_spatial is None
            if (not need_recompute_sn) and hasattr(sn_spatial, "shape") and hasattr(varr_mc, "shape"):
                try:
                    need_recompute_sn = tuple(sn_spatial.shape) != tuple(varr_mc.shape[1:])
                except Exception:
                    need_recompute_sn = True

            if need_recompute_sn:
                self.log_output.append("⚠️ sn_spatial 缺失或形状不匹配，正在自动重算...")
                sn_spatial = get_noise_fft(varr_mc.chunk({"frame": -1})).rename("sn_spatial")
                self._save_data_to_repo(sn_spatial, "sn_spatial")

            # 3) 关键修复：update_spatial 的核心维度需要单块
            # 报错: Core dimension 'f' consists of multiple chunks
            # 对应到本流程即时间维(frame)必须 rechunk 为单块
            if hasattr(varr_mc, "chunk") and "frame" in getattr(varr_mc, "dims", []):
                varr_mc = varr_mc.chunk({"frame": -1})
            if hasattr(C_init, "chunk") and "frame" in getattr(C_init, "dims", []):
                C_init = C_init.chunk({"frame": -1})

            # 4) 数值稳定性修复：清理 NaN/Inf/过大值，避免 update_spatial 内部报
            # "Input contains NaN, infinity or a value too large for dtype('float32')"
            def _sanitize_da(da: xr.DataArray, clip_abs: float = 1e6) -> xr.DataArray:
                da = xr.where(np.isfinite(da), da, np.float32(0.0))
                da = da.clip(min=-clip_abs, max=clip_abs)
                return da.astype(np.float32)

            varr_mc = _sanitize_da(varr_mc)
            A_init = _sanitize_da(A_init)
            C_init = _sanitize_da(C_init)

            # 4.1) 关键修复：保证 DataArray 具有稳定 name，避免 update_spatial 里
            # C_path = intpath/C.name.zarr/C.name 出现 shape is None
            if not getattr(varr_mc, "name", None):
                varr_mc = varr_mc.rename("varr_mc")
            if not getattr(A_init, "name", None):
                A_init = A_init.rename("A_init")
            if not getattr(C_init, "name", None):
                C_init = C_init.rename("C_init")
            if not getattr(sn_spatial, "name", None):
                sn_spatial = sn_spatial.rename("sn_spatial")

            # sn_spatial 必须为有限且正值，避免后续归一化/回归出现非法值
            sn_spatial = xr.where(np.isfinite(sn_spatial), sn_spatial, np.float32(1e-6))
            sn_spatial = xr.where(sn_spatial > 0, sn_spatial, np.float32(1e-6)).astype(np.float32)

            # 将修复后的对象写回仓库，保证后续步骤一致
            self._save_data_to_repo(varr_mc, "varr_mc")
            self._save_data_to_repo(A_init, "A_init")
            self._save_data_to_repo(C_init, "C_init")
            self._save_data_to_repo(sn_spatial, "sn_spatial")

            params = self.get_step_params(step_name)

            # 4.2) 关键修复：在 in_memory=False 时，确保 C_init 对应 zarr 已存在。
            # 否则 update_spatial 的 zarr.open_array(C_path) 会因找不到 shape 报错。
            if not params.get("in_memory", False):
                intpath = os.environ.get("MINIAN_INTERMEDIATE")
                if intpath:
                    try:
                        save_minian(C_init.rename(C_init.name or "C_init"), dpath=intpath, overwrite=True)
                    except Exception as _e:
                        self.log_output.append(f"⚠️ C_init 预落盘失败: {_e}")

            penalty_list = (
                params.get("sparse_penalty_list")
                or params.get("sparse_penal_list")
                or params.get("sparse_penalty")
                or params.get("sparse_penal")
            )
            
            # print("DEBUG:Parameters loaded successfully.")
            # print(f"DEBUG:Detailed parameters: {params}")

            if isinstance(penalty_list, (int, float)):
                penalty_list = [float(penalty_list)]
            if not penalty_list:
                penalty_list = [0.1, 0.3, 0.5, 1.0]

            dl_wnd = params.get("dl_wnd", 5)
            update_background = params.get("update_background", False)
            normalize = params.get("normalize", True)
            size_thres_raw = params.get("size_thres", (9, None))
            size_thres = self._normalize_size_thres(size_thres_raw)
            in_memory = params.get("in_memory", False)

            # 诊断：检查当前输入 footprint 面积分布，判断是否会被 size_thres 全部过滤
            try:
                area_init = (A_init > 0).sum(["height", "width"]).compute().values
                area_init = np.asarray(area_init, dtype=np.float32)
                low = size_thres[0] if isinstance(size_thres, (list, tuple)) and len(size_thres) > 0 else None
                if area_init.size > 0:
                    msg = (
                        f"[DIAG] {step_name} 输入面积统计: min={float(np.nanmin(area_init)):.2f}, "
                        f"median={float(np.nanmedian(area_init)):.2f}, max={float(np.nanmax(area_init)):.2f}, "
                        f"size_thres={size_thres}"
                    )
                    self.log_output.append(msg)
                    if low is not None:
                        keep_cnt = int((area_init > float(low)).sum())
                        self.log_output.append(
                            f"[DIAG] {step_name} 预估通过 low 阈值({low})的单元数: {keep_cnt}/{int(area_init.size)}"
                        )
            except Exception as _diag_err:
                self.log_output.append(f"[DIAG] {step_name} 面积统计失败: {_diag_err}")

            result_map = {}
            failed_penalties = []
            logs = [
                "estimating penalty parameter",
                "computing subsetting matrix",
                "fitting spatial matrix",
            ]

            for pen in penalty_list:
                self.log_output.append(f"-> 正在探索 sparse_penalty = {pen}")
                print(f"DEBUG:-> 正在探索 sparse_penalty = {pen}")

                try:
                    ret = update_spatial(
                        varr_mc,
                        A_init,
                        C_init,
                        sn_spatial,
                        b = b_init,
                        f = f_init,
                        dl_wnd=dl_wnd,
                        sparse_penal=float(pen),
                        update_background=update_background,
                        normalize=normalize,
                        size_thres=size_thres,
                        in_memory=in_memory,
                    )
                except Exception as pen_err:
                    # 兜底：磁盘模式失败时，自动用内存模式再试一次
                    if not in_memory:
                        self.log_output.append(
                            f"⚠️ sparse_penalty={pen} 磁盘模式失败，改用 in_memory=True 重试"
                        )
                        try:
                            ret = update_spatial(
                                varr_mc,
                                A_init,
                                C_init,
                                sn_spatial,
                                dl_wnd=dl_wnd,
                                sparse_penal=float(pen),
                                update_background=update_background,
                                normalize=normalize,
                                size_thres=size_thres,
                                in_memory=True,
                            )
                        except Exception as pen_err2:
                            pen_tb = traceback.format_exc()
                            msg = f"sparse_penalty={pen} 失败(含内存重试): {pen_err2}"
                            self.log_output.append("❌ " + msg)
                            self.log_output.append(pen_tb)
                            print(f"[ERROR] {step_name}::{msg}\n{pen_tb}")
                            failed_penalties.append(msg)
                            continue
                    else:
                        pen_tb = traceback.format_exc()
                        msg = f"sparse_penalty={pen} 失败: {pen_err}"
                        self.log_output.append("❌ " + msg)
                        self.log_output.append(pen_tb)
                        print(f"[ERROR] {step_name}::{msg}\n{pen_tb}")
                        failed_penalties.append(msg)
                        continue
                
                # print("DEBUG:Update spatial finished.")

                A_new = ret[0]
                mask = ret[1]
                extra = ret[2:]

                norm_fac = None
                b_new = None
                if update_background and normalize:
                    b_new, norm_fac = extra
                elif update_background:
                    b_new = extra[0]
                elif normalize:
                    norm_fac = extra[0]

                dropped = int(len(mask) - mask.sum().values)
                units_total = int(len(mask))
                log_lines = logs + [f"{dropped} out of {units_total} units dropped"]

                kept_n = int(mask.sum().values)
                if kept_n == 0:
                    # 若当前 size_thres 过严导致全部被过滤，自动放宽后重试一次
                    cur_low, cur_high = (size_thres[0], size_thres[1]) if isinstance(size_thres, (list, tuple)) and len(size_thres) >= 2 else (None, None)
                    if cur_low is not None and float(cur_low) > 1:
                        relaxed_size_thres = (1, cur_high)
                        self.log_output.append(
                            f"⚠️ sparse_penalty={pen} 过滤后无可用单元，自动放宽 size_thres={relaxed_size_thres} 重试"
                        )
                        try:
                            ret_relaxed = update_spatial(
                                varr_mc,
                                A_init,
                                C_init,
                                sn_spatial,
                                dl_wnd=dl_wnd,
                                sparse_penal=float(pen),
                                update_background=update_background,
                                normalize=normalize,
                                size_thres=relaxed_size_thres,
                                in_memory=True if not in_memory else in_memory,
                            )
                            A_new = ret_relaxed[0]
                            mask = ret_relaxed[1]
                            dropped = int(len(mask) - mask.sum().values)
                            units_total = int(len(mask))
                            log_lines = logs + [
                                f"{dropped} out of {units_total} units dropped (relaxed_size_thres={relaxed_size_thres})"
                            ]
                            kept_n = int(mask.sum().values)
                        except Exception as re_err:
                            self.log_output.append(f"❌ sparse_penalty={pen} 放宽阈值重试失败: {re_err}")

                if kept_n == 0:
                    msg = f"sparse_penalty={pen} 过滤后无可用单元(mask 全 False)"
                    self.log_output.append("❌ " + msg)
                    failed_penalties.append(msg)
                    continue

                # 给 temporal 面板准备 10 个 unit
                kept_units = A_new.coords["unit_id"].values
                sample_n = min(10, len(kept_units))
                sample_units = kept_units[:sample_n]

                A_sample = A_new.sel(unit_id=sample_units).compute()
                C_sample = C_init.sel(unit_id=sample_units).compute()

                result_map[float(pen)] = {
                    "penalty": float(pen),
                    "A_new": A_new,
                    "mask": mask,
                    "norm_fac": norm_fac,
                    "b_new": b_new,
                    "A_sample": A_sample,
                    "C_sample": C_sample,
                    "log_lines": log_lines,
                }

            if not result_map:
                raise RuntimeError(
                    "first_spatial_update_explore 所有参数探索均失败。\n" + "\n".join(failed_penalties)
                )

            ok_penalties = sorted(list(result_map.keys()))

            result = {
                "mode": "single",
                "penalty_list": ok_penalties,
                "results": result_map,
                "default_penalty": float(ok_penalties[0]),
            }

            self.exploration_results[step_name] = result
            self.steps_results[step_name] = result
            self.update_step_status(step_name, "已完成")
            return result

        except Exception as e:
            tb = traceback.format_exc()
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.log_output.append(tb)
            print(f"[ERROR] {step_name} 顶层异常: {e}\n{tb}")
            self._save_data_to_repo(
                {
                    "step": step_name,
                    "error": str(e),
                    "traceback": tb,
                },
                f"{step_name}_last_error",
            )
            self.update_step_status(step_name, "错误")
            return None

    # second_spatial_update_explore 已移除

    # 请注意： run_first_temporal_update_explore (步骤 12) 保持不变，因为它使用 Holoviews/Bokeh 风格的 visualize_temporal_update
    # 假设 create_cnmf_update_plot, compute_trace, update_temporal 等已导入

    def run_first_temporal_update_explore(self) -> dict:
        """
        初次时间更新参数探索：对给定 sparse_penalty 列表逐一执行 update_temporal，
        产出与 spatial explore 一致的可交互结果结构（single/compare）。
        """
        step_name = 'first_temporal_update_explore'
        self.update_step_status(step_name, "运行中")
        try:
            varr_mc_raw = self._load_data_from_repo('varr_mc')
            varr_mc = varr_mc_raw[1] if isinstance(varr_mc_raw, tuple) else varr_mc_raw

            A_src = self._load_data_from_repo('A_iter1')
            C_src = self._load_data_from_repo('C_iter1')
            C_chk_src = self._load_data_from_repo('C_chk_iter1')
            b_current = self._load_data_from_repo('b_iter1')
            f_current = self._load_data_from_repo('f_iter1')

            # 兼容测试链路：若 iter1 尚未生成，回退 init
            if A_src is None:
                A_src = self._load_data_from_repo('A_init')
            if C_src is None:
                C_src = self._load_data_from_repo('C_init')
            if C_chk_src is None:
                C_chk_src = C_src

            if varr_mc is None or A_src is None or C_src is None:
                raise ValueError("缺少 first_temporal_update_explore 所需输入(varr_mc/A/C)")

            required_dims = {"frame", "height", "width"}
            if not required_dims.issubset(set(getattr(varr_mc, "dims", ()))):
                raise ValueError(
                    f"varr_mc 维度异常，期望包含 {sorted(required_dims)}，实际为 {getattr(varr_mc, 'dims', None)}"
                )

            if b_current is None:
                b_current = xr.zeros_like(varr_mc.isel(frame=0, drop=True)).rename("b")
            if f_current is None:
                if {"height", "width"}.issubset(set(getattr(varr_mc, "dims", ()))):
                    f_current = xr.zeros_like(varr_mc.isel(height=0, width=0, drop=True)).rename("f")
                else:
                    # 兜底：若输入已是一维 frame 序列，则直接按其形状初始化背景时间项
                    f_current = xr.zeros_like(varr_mc).rename("f")

            if "unit_id" in A_src.coords and "unit_id" in C_src.coords:
                common_units = np.intersect1d(A_src.coords["unit_id"].values, C_src.coords["unit_id"].values)
                if len(common_units) == 0:
                    raise ValueError("A 与 C 没有公共 unit_id")
                A_src = A_src.sel(unit_id=common_units)
                C_src = C_src.sel(unit_id=common_units)
                if C_chk_src is not None and "unit_id" in C_chk_src.coords:
                    chk_units = np.intersect1d(common_units, C_chk_src.coords["unit_id"].values)
                    C_chk_src = C_chk_src.sel(unit_id=chk_units)

            params = self.get_step_params(step_name)
            penalty_list = (
                params.get('sparse_penalty_list')
                or params.get('sparse_penal_list')
                or params.get('exploration_penalties')
                or params.get('sparse_penalty')
                or params.get('sparse_penal')
            )
            if isinstance(penalty_list, (int, float)):
                penalty_list = [float(penalty_list)]
            if not penalty_list:
                penalty_list = [0.001, 0.01, 0.1]

            p = int(params.get('p', 1))
            add_lag = params.get('add_lag', 'p')
            noise_freq = float(params.get('noise_freq', 0.06))
            use_smooth = bool(params.get('use_smooth', True))

            all_units = A_src.coords["unit_id"].values
            units_to_select = min(int(params.get('sample_units', 10)), len(all_units))
            if units_to_select <= 0:
                raise ValueError("可用于 temporal explore 的 unit 数量为 0")

            np.random.seed(int(params.get('random_seed', 11)))
            sample_units = np.random.choice(all_units, units_to_select, replace=False)
            sample_units.sort()

            # 注意：temporal explore 不应就地污染前序步骤数据。
            # 这里先做深拷贝，确保后续 update_temporal 不会改写 repo 中对象。
            A_sub = A_src.sel(unit_id=sample_units).fillna(0).astype(np.float32).copy(deep=True).persist()
            C_sub = C_src.sel(unit_id=sample_units).fillna(0).astype(np.float32).copy(deep=True).persist()
            C_chk_sub = C_chk_src.sel(unit_id=sample_units).fillna(0).astype(np.float32).copy(deep=True).persist()

            # 代表性单元：默认选择 C 均值最大的单元，可通过配置 temporal_focus_unit_id 覆盖
            temporal_focus_unit = params.get('temporal_focus_unit_id', None)
            if temporal_focus_unit is None:
                c_mean = C_sub.mean(dim='frame').compute().values
                focus_idx = int(np.argmax(c_mean)) if len(c_mean) > 0 else 0
                temporal_focus_unit = int(sample_units[focus_idx])
            else:
                temporal_focus_unit = int(temporal_focus_unit)
                if temporal_focus_unit not in set(sample_units.tolist()):
                    temporal_focus_unit = int(sample_units[0])

            # 从 MC 后视频提取该单元的原始曲线（加权平均）
            A_focus = A_sub.sel(unit_id=temporal_focus_unit).fillna(0).astype(np.float32)
            w = A_focus / (A_focus.sum() + np.float32(1e-6))
            raw_mc_trace = (varr_mc * w).sum(dim=['height', 'width']).compute().values.astype(np.float32)

            self.log_output.append(f"-> temporal explore 采样单元数: {len(sample_units)}")
            self.log_output.append("-> 正在计算 YrA (残差/trace)...")
            YrA = compute_trace(varr_mc, A_sub, b_current, C_chk_sub, f_current).copy(deep=True).persist()

            result_map = {}
            failed_penalties = []
            for pen in penalty_list:
                self.log_output.append(f"-> 正在探索 temporal sparse_penalty = {pen}")
                try:
                    # 每个 penalty 使用独立副本，防止一个参数运行后影响下一个参数
                    A_in = A_sub.copy(deep=True)
                    C_in = C_sub.copy(deep=True)
                    YrA_in = YrA.copy(deep=True)

                    cur_C, cur_S, cur_b0, cur_c0, cur_g, cur_mask = update_temporal(
                        A_in,
                        C_in,
                        YrA=YrA_in,
                        sparse_penal=float(pen),
                        p=p,
                        use_smooth=use_smooth,
                        add_lag=add_lag,
                        noise_freq=noise_freq,
                    )
                except Exception as pen_err:
                    # 二次尝试：使用更保守参数，避免单个参数类型/数值导致全失败
                    try:
                        cur_C, cur_S, cur_b0, cur_c0, cur_g, cur_mask = update_temporal(
                            A_in,
                            C_in,
                            YrA=YrA_in,
                            sparse_penal=float(pen),
                            p=max(1, p),
                            use_smooth=False,
                            add_lag='p',
                            noise_freq=noise_freq,
                        )
                        self.log_output.append(f"⚠️ sparse_penalty={pen} 首次失败，已用保守参数重试成功")
                    except Exception as pen_err_2:
                        msg = f"sparse_penalty={pen} 失败: {pen_err}; retry失败: {pen_err_2}"
                        self.log_output.append("❌ " + msg)
                        failed_penalties.append(msg)
                        continue

                # update_temporal 可能筛掉部分 unit，直接 sel 会触发 KeyError。
                # 这里统一重建到 sample_units 轴，缺失单元以 0 填充，保证可视化结构稳定。
                C_show = cur_C.reindex(unit_id=sample_units).fillna(np.float32(0)).compute().astype(np.float32)
                S_show = cur_S.reindex(unit_id=sample_units).fillna(np.float32(0)).compute().astype(np.float32)
                C_ref = C_sub.compute().astype(np.float32)
                mask_kept = int(cur_mask.sum().values) if hasattr(cur_mask, "sum") else len(sample_units)

                if "unit_id" in cur_C.coords and temporal_focus_unit in set(cur_C.coords["unit_id"].values.tolist()):
                    focus_uid = int(temporal_focus_unit)
                elif "unit_id" in cur_C.coords and cur_C.sizes.get("unit_id", 0) > 0:
                    focus_uid = int(cur_C.coords["unit_id"].values[0])
                    self.log_output.append(
                        f"⚠️ sparse_penalty={pen} 下 focus unit {temporal_focus_unit} 被筛除，已回退到 {focus_uid}"
                    )
                else:
                    focus_uid = int(temporal_focus_unit)

                c_after_trace = cur_C.reindex(unit_id=[focus_uid]).fillna(np.float32(0)).isel(unit_id=0).compute().values.astype(np.float32)
                fit_trace = (
                    (cur_C + cur_b0 + cur_c0)
                    .reindex(unit_id=[focus_uid])
                    .fillna(np.float32(0))
                    .isel(unit_id=0)
                    .compute()
                    .values
                    .astype(np.float32)
                )
                spike_trace = cur_S.reindex(unit_id=[focus_uid]).fillna(np.float32(0)).isel(unit_id=0).compute().values.astype(np.float32)

                result_map[float(pen)] = {
                    "penalty": float(pen),
                    "C_before": C_ref,
                    "C_after": C_show,
                    "S_after": S_show,
                    "focus_unit": int(focus_uid),
                    "c_after_trace": c_after_trace,
                    "raw_mc_trace": raw_mc_trace,
                    "fit_trace": fit_trace,
                    "spike_trace": spike_trace,
                    "sample_units": sample_units.tolist(),
                    "log_lines": [
                        f"p={p}",
                        f"add_lag={add_lag}",
                        f"noise_freq={noise_freq}",
                        f"use_smooth={use_smooth}",
                        f"kept_units={mask_kept}",
                    ],
                }

            if not result_map:
                self.log_output.append("⚠️ temporal explore 所有参数失败，回退为原始 C 曲线可视化结果。")
                c_ref = C_sub.compute().astype(np.float32)
                uid0 = int(sample_units[0])
                c_trace = c_ref.sel(unit_id=uid0).values.astype(np.float32)
                z = np.zeros_like(c_trace, dtype=np.float32)
                for pen in penalty_list:
                    result_map[float(pen)] = {
                        "penalty": float(pen),
                        "C_before": c_ref,
                        "C_after": c_ref,
                        "S_after": xr.zeros_like(c_ref),
                        "focus_unit": uid0,
                        "c_after_trace": c_trace,
                        "raw_mc_trace": c_trace,
                        "fit_trace": c_trace,
                        "spike_trace": z,
                        "sample_units": sample_units.tolist(),
                        "log_lines": [
                            "fallback=true",
                            f"failed_count={len(failed_penalties)}",
                        ],
                    }

            ok_penalties = sorted(list(result_map.keys()))
            result = {
                "mode": "single",
                "penalty_list": ok_penalties,
                "results": result_map,
                "default_penalty": float(ok_penalties[0]),
            }

            self.exploration_results[step_name] = result
            self.steps_results[step_name] = result
            self.update_step_status(step_name, "已完成")
            return result

        except Exception as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.log_output.append(traceback.format_exc())
            self.update_step_status(step_name, "错误")
            return None

    # second_temporal_update_explore 已移除

    def _resolve_temporal_exec_penalty(self, explore_step_name: str, params: dict) -> float:
        exp_data = self.get_exploration_result(explore_step_name)
        exp_state = self.get_exploration_state(explore_step_name)

        if exp_data and exp_data.get("results"):
            keys = sorted([float(k) for k in exp_data["results"].keys()])
            selected = exp_state.get("selected_penalty", exp_data.get("default_penalty", keys[0]))
            try:
                selected = float(selected)
                return min(keys, key=lambda x: abs(x - selected))
            except Exception:
                return float(keys[0])

        p = (
            params.get("sparse_penalty")
            or params.get("sparse_penal")
            or params.get("sparse_penalty_list")
            or params.get("sparse_penal_list")
            or params.get("exploration_penalties")
            or 0.1
        )
        if isinstance(p, (list, tuple)) and len(p) > 0:
            try:
                return float(p[0])
            except Exception:
                return 0.1
        try:
            return float(str(p))
        except Exception:
            return 0.1

    def _run_temporal_update_exec_common(
        self,
        step_name: str,
        explore_step_name: str,
        a_key: str,
        c_key: str,
        c_chk_key: str,
        b_key: str,
        f_key: str,
        vis_step_title: str,
        output_a_mrg_key: Optional[str] = None,
        output_c_mrg_key: Optional[str] = None,
        output_c_chk_mrg_key: Optional[str] = None,
        output_sig_mrg_key: Optional[str] = None,
    ) -> bool:
        self.update_step_status(step_name, "运行中")
        try:
            intpath = os.environ.get("MINIAN_INTERMEDIATE", "./intermediate_data")
            diag_bundle: Dict[str, Any] = {}

            varr_mc_raw = self._load_data_from_repo("varr_mc")
            varr_mc = varr_mc_raw[1] if isinstance(varr_mc_raw, tuple) else varr_mc_raw
            A_current = self._load_data_from_repo(a_key)
            C_current = self._load_data_from_repo(c_key)
            C_chk_current = self._load_data_from_repo(c_chk_key)
            b_current = self._load_data_from_repo(b_key)
            f_current = self._load_data_from_repo(f_key)

            if varr_mc is None or A_current is None or C_current is None:
                raise ValueError(f"缺少 temporal update 输入(varr_mc/{a_key}/{c_key})")

            # 与 explore 一致的预处理：frame 单块 + 数值清理
            varr_mc = varr_mc.chunk({"frame": -1}).fillna(0).astype(np.float32)
            A_current = A_current.fillna(0).astype(np.float32)
            C_current = C_current.chunk({"frame": -1}).fillna(0).astype(np.float32)

            if C_chk_current is None:
                C_chk_current = C_current
            C_chk_current = C_chk_current.chunk({"frame": -1}).fillna(0).astype(np.float32)

            if "unit_id" in A_current.coords and "unit_id" in C_current.coords:
                common_units = np.intersect1d(A_current.coords["unit_id"].values, C_current.coords["unit_id"].values)
                if len(common_units) == 0:
                    raise ValueError(f"{a_key}/{c_key} 无公共 unit_id")
                A_current = A_current.sel(unit_id=common_units)
                C_current = C_current.sel(unit_id=common_units)

            # 详细诊断：temporal update 输入质量
            diag_bundle["A_current_before_temporal"] = self._diagnose_da(
                A_current, f"{step_name}:A_current_before_temporal", per_unit=True
            )
            diag_bundle["C_current_before_temporal"] = self._diagnose_da(
                C_current, f"{step_name}:C_current_before_temporal", per_unit=True
            )

            # C_chk 必须与 C_current 的 unit 轴严格一致，否则 compute_trace 会失败
            if "unit_id" in C_chk_current.coords:
                C_chk_current = (
                    C_chk_current
                    .reindex(unit_id=C_current.coords["unit_id"].values)
                    .fillna(np.float32(0))
                    .astype(np.float32)
                )

            b_template = varr_mc.isel(frame=0, drop=True).rename("b")
            if b_current is None or not {"height", "width"}.issubset(set(getattr(b_current, "dims", ()))):
                b_current = xr.zeros_like(b_template)
            else:
                b_current = b_current.reindex_like(b_template).fillna(np.float32(0)).astype(np.float32).rename("b")

            f_template = varr_mc.isel(height=0, width=0, drop=True).rename("f")
            if f_current is None or "frame" not in set(getattr(f_current, "dims", ())):
                f_current = xr.zeros_like(f_template)
            else:
                f_current = (
                    f_current
                    .reindex(frame=f_template.coords["frame"])
                    .fillna(np.float32(0))
                    .astype(np.float32)
                    .rename("f")
                )

            params = dict(self.get_step_params(explore_step_name) or {})
            exec_params = self.get_step_params(step_name)
            params.update(exec_params if exec_params else {})

            sparse_pen = self._resolve_temporal_exec_penalty(explore_step_name, params)
            p = int(params.get("p", 1))
            add_lag = params.get("add_lag", "p")
            noise_freq = float(params.get("noise_freq", 0.06))
            use_smooth = bool(params.get("use_smooth", True))

            merge_kwargs = params.get("merge_kwargs") or {}

            self.log_output.append(f"-> {step_name} 使用 sparse_penalty={sparse_pen}")
            self.log_output.append("-> 正在计算 YrA (残差/trace)...")
            YrA = compute_trace(varr_mc, A_current, b_current, C_chk_current, f_current).persist()

            YrA_saved = save_minian(
                YrA.rename("YrA"),
                intpath,
                overwrite=True,
                chunks={"unit_id": 1, "frame": -1},
            )
            self._save_data_to_repo(YrA_saved, f"{step_name}_YrA")

            self.log_output.append("-> 正在执行 temporal update...")
            try:
                C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
                    A_current,
                    C_current,
                    YrA=YrA,
                    sparse_penal=float(sparse_pen),
                    p=p,
                    use_smooth=use_smooth,
                    add_lag=add_lag,
                    noise_freq=noise_freq,
                )
            except Exception:
                C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
                    A_current,
                    C_current,
                    YrA=YrA,
                    sparse_penal=float(sparse_pen),
                    p=max(1, p),
                    use_smooth=False,
                    add_lag="p",
                    noise_freq=noise_freq,
                )
                
            print(f"Sum of C_new: {float(np.abs(C_new).sum().compute().values)}")

            # 详细诊断：temporal update 输出质量
            diag_bundle["C_new_after_temporal"] = self._diagnose_da(
                C_new, f"{step_name}:C_new_after_temporal", per_unit=True
            )
            diag_bundle["S_new_after_temporal"] = self._diagnose_da(
                S_new, f"{step_name}:S_new_after_temporal", per_unit=True
            )

            if C_new.sizes.get("unit_id", 0) <= 0:
                raise ValueError("temporal update 后无可用 unit（全部被丢弃）")

            C_new = C_new.rename("C_new")

            all_units = np.asarray(C_current.coords["unit_id"].values)
            kept_units = np.asarray(C_new.coords["unit_id"].values)
            dropped_units = np.setdiff1d(all_units, kept_units)

            dropped_before = []
            dropped_after = []
            dropped_ids = []
            n_show = min(3, len(dropped_units))
            for uid in dropped_units[:n_show]:
                b_trace = C_current.sel(unit_id=uid).compute().values.astype(np.float32)
                dropped_before.append(b_trace)
                dropped_after.append(np.zeros_like(b_trace, dtype=np.float32))
                dropped_ids.append(int(uid))

            self.log_output.append("-> 正在生成 temporal update 热图可视化结果...")
            img_array_update = create_temporal_matrix_plot(
                C_current.compute().astype(np.float32).values,
                C_new.compute().astype(np.float32).values,
                S_new.compute().astype(np.float32).values,
                step_name=f"{vis_step_title} Update",
                dropped_examples={
                    "unit_ids": dropped_ids,
                    "before": dropped_before,
                    "after": dropped_after,
                },
            )
            self._save_data_to_repo(img_array_update, f"{step_name}_update_vis_array")
            self._save_data_to_repo(img_array_update, f"{step_name}_c_s_vis_array")

            chk = self._load_data_from_repo("chk_settings")
            if not isinstance(chk, dict):
                chk = {"frame": -1}
            chk_frame = chk.get("frame", -1)
            try:
                chk_frame = int(chk_frame)
            except Exception:
                chk_frame = -1

            self.log_output.append("-> 正在保存 temporal update 关键结果矩阵...")
            C_saved = save_minian(
                C_new.rename("C").chunk({"unit_id": 1, "frame": -1}),
                intpath,
                overwrite=True,
            )
            C_chk_saved = save_minian(
                C_saved.rename("C_chk"),
                intpath,
                overwrite=True,
                chunks={"unit_id": -1, "frame": chk_frame},
            )
            S_saved = save_minian(
                S_new.rename("S").chunk({"unit_id": 1, "frame": -1}),
                intpath,
                overwrite=True,
            )
            b0_saved = save_minian(
                b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}),
                intpath,
                overwrite=True,
            )
            c0_saved = save_minian(
                c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}),
                intpath,
                overwrite=True,
            )

            self._save_data_to_repo(C_saved, "C_tmp_merge")
            self._save_data_to_repo(C_chk_saved, "C_chk_tmp_merge")
            self._save_data_to_repo(S_saved, "S_tmp_merge")
            self._save_data_to_repo(b0_saved, "b0_tmp_merge")
            self._save_data_to_repo(c0_saved, "c0_tmp_merge")
            # print(f"DEBUG:Sum of C_saved: {float(np.abs(C_saved).sum().compute().values)}")
            A_for_merge = A_current.sel(unit_id=C_saved.coords["unit_id"].values)
            self._save_data_to_repo(A_for_merge, "A_tmp_merge")
            diag_bundle["A_for_merge_before_unit_merge"] = self._diagnose_da(
                A_for_merge, f"{step_name}:A_for_merge_before_unit_merge", per_unit=True
            )

            self.log_output.append("-> 正在执行 unit merge...")
            A_mrg, C_mrg, sig_mrg_list = unit_merge(
                A_for_merge,
                C_saved,
                [C_saved + b0_saved + c0_saved],
                **merge_kwargs,
            )
            sig_mrg = sig_mrg_list[0] if sig_mrg_list else (C_mrg + 0)

            # 同步 merge 后 unit 轴到其它 temporal 矩阵，避免仅 C 更新导致维度不一致。
            merged_units = C_mrg.coords["unit_id"].values if "unit_id" in C_mrg.coords else None
            if merged_units is not None:
                S_mrg = (
                    S_saved.reindex(unit_id=merged_units)
                    .fillna(np.float32(0))
                    .astype(np.float32)
                    .rename("S")
                )
                b0_mrg = (
                    b0_saved.reindex(unit_id=merged_units)
                    .fillna(np.float32(0))
                    .astype(np.float32)
                    .rename("b0")
                )
                c0_mrg = (
                    c0_saved.reindex(unit_id=merged_units)
                    .fillna(np.float32(0))
                    .astype(np.float32)
                    .rename("c0")
                )
                self._save_data_to_repo(S_mrg, "S_tmp_merge")
                self._save_data_to_repo(b0_mrg, "b0_tmp_merge")
                self._save_data_to_repo(c0_mrg, "c0_tmp_merge")
                self.log_output.append(
                    f"-> merge 后矩阵同步: C/S/b0/c0 unit 数 {int(C_saved.sizes.get('unit_id', 0))} -> {int(C_mrg.sizes.get('unit_id', 0))}"
                )
            else:
                S_mrg = S_saved
                b0_mrg = b0_saved
                c0_mrg = c0_saved
                
                # print(f"DEBUG:Sum of C_merged: {float(np.abs(C_mrg).sum().compute().values)}")

            img_array_merge = create_merge_matrix_plot(
                C_saved.compute().astype(np.float32).values,
                C_mrg.compute().astype(np.float32).values,
                step_name=f"{vis_step_title} Merge",
            )
            self._save_data_to_repo(img_array_merge, f"{step_name}_merge_vis_array")

            if output_a_mrg_key and output_c_mrg_key and output_c_chk_mrg_key and output_sig_mrg_key:
                    # print(f"DEBUG:Sum of C_mrg for saving: {float(np.abs(C_mrg).sum().compute().values)}")
                    # 关键修复：
                    # 1) A 主键保持 dask-backed，避免 second spatial 中 map_blocks 在 numpy 后端崩溃
                    # 2) C 先实体化，切断对上游惰性图/被 overwrite 路径的依赖，避免 C 退化为全零
                    A_mem = A_mrg.compute().rename("A")
                    C_mem = C_mrg.compute().rename("C")
                    sig_mem = sig_mrg.rename("sig")

                    print(f"DEBUG: Sum of C_mem in memory: {float(np.abs(C_mem).sum())}")

                    # 运行态主键（供后续步骤直接消费）
                    self._save_data_to_repo(A_mem, output_a_mrg_key)
                    self._save_data_to_repo(C_mem, output_c_mrg_key)
                    self._save_data_to_repo(C_mem.rename("C_chk"), output_c_chk_mrg_key)
                    self._save_data_to_repo(sig_mem, output_sig_mrg_key)
                    self._save_data_to_repo(S_mrg, "S_iter1_merged")
                    self._save_data_to_repo(b0_mrg, "b0_iter1_merged")
                    self._save_data_to_repo(c0_mrg, "c0_iter1_merged")

                    # 落盘副本（排查/持久化）
                    # 仅作为辅助，不应影响主流程；失败时记录告警并继续。
                    try:
                        A_final = save_minian(A_mrg.rename("A"), intpath, overwrite=True)
                        C_final = save_minian(C_mem.rename("C"), intpath, overwrite=True)
                        C_chk_final = save_minian(C_final.rename("C_chk"), intpath, overwrite=True)
                        sig_final = save_minian(sig_mrg.rename("sig"), intpath, overwrite=True)

                        self._save_data_to_repo(A_final, f"{output_a_mrg_key}_disk")
                        self._save_data_to_repo(C_final, f"{output_c_mrg_key}_disk")
                        self._save_data_to_repo(C_chk_final, f"{output_c_chk_mrg_key}_disk")
                        self._save_data_to_repo(sig_final, f"{output_sig_mrg_key}_disk")
                    except Exception as disk_err:
                        self.log_output.append(
                            f"⚠️ {step_name} 落盘副本保存失败（不影响主流程）: {disk_err}"
                        )

                    # 备用镜像键（诊断/回退）
                    self._save_data_to_repo(A_mem, f"{output_a_mrg_key}_mem")
                    self._save_data_to_repo(C_mem, f"{output_c_mrg_key}_mem")

                    # # 给 second spatial 专门留一份可追溯诊断
                    # if step_name == "first_temporal_update_exec":
                    #     self._save_data_to_repo(diag_bundle, "first_temporal_post_diag_detail")
                    #     hint = ""
                    #     a_diag = diag_bundle.get("A_mrg_after_unit_merge", {})
                    #     if a_diag.get("available"):
                    #         if int(a_diag.get("units_abs_gt0", 0)) == 0:
                    #             hint = "first_temporal 合并后 A 全零/近零，second spatial 会全 drop"
                    #         elif int(a_diag.get("units_pos_gt0", 0)) == 0:
                    #             hint = "first_temporal 合并后 A 无正值面积，second spatial 基于正面积筛选会全 drop"
                    #     if hint:
                    #         self.log_output.append(f"[DIAG] {step_name} root_cause_hint: {hint}")

            self.log_output.append("✅ temporal update + merge 完成，并已保存关键矩阵。")

            self.steps_results[step_name] = True
            self.update_step_status(step_name, "已完成")
            return True

        except Exception as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.log_output.append(traceback.format_exc())
            self.update_step_status(step_name, "错误")
            return False

    def run_first_temporal_update_exec(self) -> bool:
        return self._run_temporal_update_exec_common(
            step_name="first_temporal_update_exec",
            explore_step_name="first_temporal_update_explore",
            a_key="A_iter1",
            c_key="C_iter1",
            c_chk_key="C_chk_iter1",
            b_key="b_iter1",
            f_key="f_iter1",
            vis_step_title="First",
            output_a_mrg_key="A_iter1_merged",
            output_c_mrg_key="C_iter1_merged",
            output_c_chk_mrg_key="C_chk_iter1_merged",
            output_sig_mrg_key="sig_iter1_merged",
        )

    # 兼容旧测试入口
    def run_first_temporal_update(self) -> bool:
        return self.run_first_temporal_update_exec()
    
    def _resolve_spatial_exec_penalty(self, explore_step_name: str, explore_params: dict, exec_params: dict) -> float:
        # 1) 执行步骤显式参数优先（用于“确认最终值”）
        p_exec = exec_params.get("sparse_penalty", exec_params.get("sparse_penal", None))
        if p_exec is not None:
            try:
                return float(p_exec)
            except Exception:
                pass

        exp_data = self.get_exploration_result(explore_step_name)
        exp_state = self.get_exploration_state(explore_step_name)

        if exp_data and exp_data.get("results"):
            keys = sorted([float(k) for k in exp_data["results"].keys()])
            selected = exp_state.get("selected_penalty", exp_data.get("default_penalty", keys[0]))
            try:
                selected = float(selected)
                return min(keys, key=lambda x: abs(x - selected))
            except Exception:
                return float(keys[0])

        p = (
            explore_params.get("sparse_penalty")
            or explore_params.get("sparse_penal")
            or explore_params.get("sparse_penalty_list")
            or explore_params.get("sparse_penal_list")
            or 0.1
        )
        if isinstance(p, (list, tuple)) and len(p) > 0:
            try:
                return float(p[0])
            except Exception:
                return 0.1
        try:
            return float(str(p))
        except Exception:
            return 0.1

    def _prepare_spatial_exec_inputs(self, a_key: str, c_key: str):
        varr_mc_raw = self._load_data_from_repo("varr_mc")
        varr_mc = varr_mc_raw[1] if isinstance(varr_mc_raw, tuple) else varr_mc_raw
        A_init = self._load_data_from_repo(a_key)
        C_init = self._load_data_from_repo(c_key)
        sn_spatial = self._load_data_from_repo("sn_spatial")

        if varr_mc is None or A_init is None or C_init is None:
            raise ValueError(f"缺少空间更新执行所需输入(varr_mc/{a_key}/{c_key})")

        if "unit_id" in A_init.coords and "unit_id" in C_init.coords:
            common_units = np.intersect1d(A_init.coords["unit_id"].values, C_init.coords["unit_id"].values)
            if len(common_units) == 0:
                raise ValueError(f"{a_key} 与 {c_key} 没有公共 unit_id")
            A_init = A_init.sel(unit_id=common_units)
            C_init = C_init.sel(unit_id=common_units)

        need_recompute_sn = sn_spatial is None
        if (not need_recompute_sn) and hasattr(sn_spatial, "shape") and hasattr(varr_mc, "shape"):
            try:
                need_recompute_sn = tuple(sn_spatial.shape) != tuple(varr_mc.shape[1:])
            except Exception:
                need_recompute_sn = True
        if need_recompute_sn:
            sn_spatial = get_noise_fft(varr_mc.chunk({"frame": -1})).rename("sn_spatial")

        varr_mc = varr_mc.chunk({"frame": -1}).fillna(0).astype(np.float32)
        C_init = C_init.chunk({"frame": -1}).fillna(0).astype(np.float32)
        A_init = A_init.fillna(0).astype(np.float32)
        sn_spatial = xr.where(np.isfinite(sn_spatial), sn_spatial, np.float32(1e-6))
        sn_spatial = xr.where(sn_spatial > 0, sn_spatial, np.float32(1e-6)).astype(np.float32)

        return varr_mc, A_init, C_init, sn_spatial

    def _run_spatial_update_exec_common(
        self,
        step_name: str,
        explore_step_name: str,
        a_key: str,
        c_key: str,
        f_fallback_key: str,
        output_a_key: str,
        output_c_key: str,
        output_c_chk_key: str,
        output_b_key: str,
        output_f_key: str,
        vis_step_title: str,
        save_chk_settings: bool = False,
    ) -> bool:
        self.update_step_status(step_name, "运行中")
        try:
            intpath = os.environ.get("MINIAN_INTERMEDIATE", "./intermediate_data")

            varr_mc, A_init, C_init, sn_spatial = self._prepare_spatial_exec_inputs(a_key, c_key)
            
            if step_name == "first_spatial_update_exec":
                b_in = self._load_data_from_repo("b_init")
                f_in = self._load_data_from_repo("f_init")
            else:
                b_in = self._load_data_from_repo("b_iter1")
                f_in = self._load_data_from_repo("f_iter1")

            explore_params = self.get_step_params(explore_step_name) or {}
            exec_params = self.get_step_params(step_name) or {}
            params = dict(explore_params)
            params.update(exec_params)

            sparse_pen = self._resolve_spatial_exec_penalty(explore_step_name, explore_params, exec_params)
            self.log_output.append(f"-> {step_name} 使用 sparse_penalty={sparse_pen}")

            size_thres_arg = self._normalize_size_thres(params.get("size_thres", (9, None)))
            ret = update_spatial(
                varr_mc,
                A_init,
                C_init,
                sn_spatial,
                b = b_in,
                f = f_in,
                dl_wnd=params.get("dl_wnd", 5),
                sparse_penal=float(sparse_pen),
                update_background=params.get("update_background", False),
                normalize=params.get("normalize", True),
                size_thres=size_thres_arg,
                in_memory=params.get("in_memory", False),
            )

            A_new = ret[0]
            mask = ret[1]
            extra = ret[2:]

            # 关键兜底：若执行阶段被阈值全部过滤，自动放宽 size_thres 重试一次
            kept_n = int(mask.sum().values) if hasattr(mask, "sum") else 0
            if kept_n == 0:
                relaxed_size_thres = (1, None)
                self.log_output.append(
                    f"⚠️ {step_name} 在 size_thres={size_thres_arg} 下全部单元被过滤，"
                    f"自动放宽到 size_thres={relaxed_size_thres} 重试一次"
                )
                ret = update_spatial(
                    varr_mc,
                    A_init,
                    C_init,
                    sn_spatial,
                    dl_wnd=params.get("dl_wnd", 5),
                    sparse_penal=float(sparse_pen),
                    update_background=params.get("update_background", False),
                    normalize=params.get("normalize", True),
                    size_thres=relaxed_size_thres,
                    in_memory=True,
                )
                A_new = ret[0]
                mask = ret[1]
                extra = ret[2:]

                kept_n = int(mask.sum().values) if hasattr(mask, "sum") else 0
                if kept_n == 0:
                    raise ValueError(
                        f"{step_name} 过滤后无可用单元（size_thres={size_thres_arg}，放宽后仍为0）"
                    )

            norm_fac = None
            b_new = None
            if params.get("update_background", False) and params.get("normalize", True):
                b_new, norm_fac = extra
            elif params.get("update_background", False):
                b_new = extra[0]
            elif params.get("normalize", True):
                norm_fac = extra[0]

            C_new = C_init.sel(unit_id=mask)
            if norm_fac is not None:
                C_new = C_new * norm_fac
            C_chk_new = C_new.rename("C_chk")

            if b_new is None:
                b_new, f_new = update_background(varr_mc, A_new, C_chk_new)
            else:
                f_new = self._load_data_from_repo(f_fallback_key)
                if f_new is None:
                    f_new = xr.zeros_like(C_new.isel(unit_id=0, drop=True)).rename("f")

            A_init_max = A_init.max("unit_id").compute().astype(np.float32).values
            A_init_sum = (A_init.fillna(0) > 0).sum("unit_id").compute().astype(np.uint8).values
            A_new_max = A_new.max("unit_id").compute().astype(np.float32).values
            A_new_sum = (A_new > 0).sum("unit_id").compute().astype(np.uint8).values
            img_array = create_spatial_update_plot(
                A_init_max,
                A_init_sum,
                A_new_max,
                A_new_sum,
                step_name=vis_step_title,
            )
            self._save_data_to_repo(img_array, f"{step_name}_vis_array")

            chk = self._load_data_from_repo("chk_settings")
            if not isinstance(chk, dict):
                chk = {"frame": -1}
            chk_frame = chk.get("frame", -1)
            try:
                chk_frame = int(chk_frame)
            except Exception:
                chk_frame = -1

            # 关键结果矩阵保存（对齐 Minian notebook 的 save results 逻辑）
            A = save_minian(
                A_new.rename("A"),
                intpath,
                overwrite=True,
                chunks={"unit_id": 1, "height": -1, "width": -1},
            )
            b = save_minian(b_new.rename("b"), intpath, overwrite=True)
            f = save_minian(
                f_new.chunk({"frame": chk_frame}).rename("f"),
                intpath,
                overwrite=True,
            )
            C = save_minian(C_new.rename("C"), intpath, overwrite=True)
            C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

            self._save_data_to_repo(A, output_a_key)
            self._save_data_to_repo(C, output_c_key)
            self._save_data_to_repo(C_chk, output_c_chk_key)
            self._save_data_to_repo(b, output_b_key)
            self._save_data_to_repo(f, output_f_key)
            if save_chk_settings:
                self._save_data_to_repo({"frame": -1}, "chk_settings")

            self.log_output.append("✅ spatial update 关键矩阵已保存: A/b/f/C/C_chk")

            self.steps_results[step_name] = True
            self.update_step_status(step_name, "已完成")
            return True

        except Exception as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.log_output.append(traceback.format_exc())
            self.update_step_status(step_name, "错误")
            return False

    def run_first_spatial_update_exec(self) -> bool:
        return self._run_spatial_update_exec_common(
            step_name="first_spatial_update_exec",
            explore_step_name="first_spatial_update_explore",
            a_key="A_init",
            c_key="C_init",
            f_fallback_key="f_init",
            output_a_key="A_iter1",
            output_c_key="C_iter1",
            output_c_chk_key="C_chk_iter1",
            output_b_key="b_iter1",
            output_f_key="f_iter1",
            vis_step_title="First Update",
            save_chk_settings=True,
        )

    # second_spatial_update / second_temporal_update 已移除

    def _resolve_save_data_inputs(self) -> Dict[str, Optional[xr.DataArray]]:
        """按优先级解析可用于 save_data 的矩阵来源。"""
        intpath = os.environ.get("MINIAN_INTERMEDIATE", "./intermediate_data")
        disk_ds_dict: Dict[str, Any] = {}

        try:
            if os.path.isdir(intpath):
                loaded = open_minian(intpath, return_dict=True)
                if isinstance(loaded, dict):
                    disk_ds_dict = loaded
                elif isinstance(loaded, xr.Dataset):
                    disk_ds_dict = {str(k): loaded[k] for k in loaded.data_vars.keys()}
                else:
                    disk_ds_dict = {}
        except Exception:
            # 读取磁盘失败时保持静默，继续走内存仓库
            disk_ds_dict = {}

        def _valid(name: str, v: Any) -> bool:
            if v is None:
                return False
            if not isinstance(v, xr.DataArray):
                return True
            try:
                if name in ("A", "C", "S", "C_chk", "sig", "YrA") and "unit_id" in v.dims:
                    if int(v.sizes.get("unit_id", 0)) <= 0:
                        return False
                if name in ("C", "S", "C_chk", "sig", "f", "YrA") and "frame" in v.dims:
                    if int(v.sizes.get("frame", 0)) <= 0:
                        return False
            except Exception:
                return True
            return True

        def _pick(name: str, *keys):
            for k in keys:
                v = self._load_data_from_repo(k)
                if _valid(name, v):
                    return v
            # 回退：从中间目录读取（文件名通常直接是 A/C/S/b/f/...）
            if name in disk_ds_dict:
                v = disk_ds_dict.get(name)
                if _valid(name, v):
                    return v
            return None

        return {
            "A": _pick("A", "A_iter1_merged", "A_iter1", "A_init"),
            "C": _pick("C", "C_iter1_merged", "C_iter1", "C_init"),
            "S": _pick("S", "S_iter1_merged", "S_tmp_merge"),
            "YrA": _pick("YrA", "first_temporal_update_exec_YrA", "YrA"),
            "c0": _pick("c0", "c0_iter1_merged", "c0_tmp_merge"),
            "b0": _pick("b0", "b0_iter1_merged", "b0_tmp_merge"),
            "b": _pick("b", "b_iter1", "b_init"),
            "f": _pick("f", "f_iter1", "f_init"),
            "C_chk": _pick("C_chk", "C_chk_iter1_merged", "C_chk_iter1", "C_chk_tmp_merge"),
            "sig": _pick("sig", "sig_iter1_merged"),
        }

    @staticmethod
    def _apply_excluded_units(data: Optional[xr.DataArray], excluded_units: List[int]) -> Optional[xr.DataArray]:
        if data is None:
            return None
        if not excluded_units or "unit_id" not in getattr(data, "dims", ()): 
            return data
        keep_units = [
            int(u) for u in data.coords["unit_id"].values
            if int(u) not in set(int(x) for x in excluded_units)
        ]
        if len(keep_units) == 0:
            raise ValueError("排除 unit_id 后无可保存单元，请减少排除数量")
        return data.sel(unit_id=keep_units)

    def run_save_data(self) -> bool:
        """
        步骤 20: 数据保存
        支持：矩阵多选 + 多保存格式(zarr/netcdf/csv/npy) + unit_id 排除过滤。
        """
        step_name = 'save_data'
        self.update_step_status(step_name, "运行中")

        try:
            params = self.get_step_params(step_name)

            prev_saved_paths = self._load_data_from_repo("save_data_saved_paths")
            if isinstance(prev_saved_paths, dict) and len(prev_saved_paths) > 0:
                self.log_output.append("-> 检测到已有保存记录，本次将覆盖同名输出并更新保存记录。")

            selected_matrices = params.get("selected_matrices", ["A", "C", "S", "YrA", "c0", "b0", "b", "f"])
            if not isinstance(selected_matrices, list):
                selected_matrices = ["A", "C", "S", "YrA", "c0", "b0", "b", "f"]

            save_format = str(params.get("save_format", "zarr")).lower()
            output_dir = str(params.get("output_dir", "./minian_output"))
            excluded_units_raw = params.get("excluded_unit_ids", [])
            excluded_units = []
            if isinstance(excluded_units_raw, list):
                for x in excluded_units_raw:
                    try:
                        excluded_units.append(int(x))
                    except Exception:
                        continue

            os.makedirs(output_dir, exist_ok=True)

            available = self._resolve_save_data_inputs()
            to_save: Dict[str, xr.DataArray] = {}
            for name in selected_matrices:
                if name not in available:
                    self.log_output.append(f"⚠️ 跳过未知矩阵: {name}")
                    continue
                arr = self._apply_excluded_units(available.get(name), excluded_units)
                if arr is None:
                    self.log_output.append(f"⚠️ 跳过缺失矩阵: {name}")
                    continue
                to_save[name] = arr.rename(name)

            if not to_save:
                raise ValueError("没有可保存的矩阵，请检查选择项与上游步骤输出")

            self.log_output.append(
                f"-> 正在保存矩阵 {list(to_save.keys())}，格式={save_format}，目录={output_dir}"
            )

            saved_paths: Dict[str, str] = {}
            if save_format == "zarr":
                for name, arr in to_save.items():
                    saved = save_minian(arr, dpath=output_dir, overwrite=True)
                    saved_paths[name] = str(getattr(saved, "name", name))
            elif save_format == "netcdf":
                ds = xr.Dataset({k: v for k, v in to_save.items()})
                out_file = os.path.join(output_dir, "minian_dataset.nc")
                ds.to_netcdf(out_file)
                saved_paths["dataset"] = out_file
            elif save_format == "csv":
                for name, arr in to_save.items():
                    out_file = os.path.join(output_dir, f"{name}.csv")
                    arr.rename(name).to_series().reset_index().to_csv(out_file, index=False)
                    saved_paths[name] = out_file
            elif save_format == "npy":
                for name, arr in to_save.items():
                    out_file = os.path.join(output_dir, f"{name}.npy")
                    np.save(out_file, arr.values)
                    saved_paths[name] = out_file
            else:
                raise ValueError(f"不支持的保存格式: {save_format}")

            self._save_data_to_repo(saved_paths, "save_data_saved_paths")
            self._save_data_to_repo(excluded_units, "save_data_excluded_units")

            self.log_output.append("✅ 数据保存完成。")
            self.update_step_status(step_name, "已完成")
            return True
            
        except Exception as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.log_output.append(traceback.format_exc())
            self.update_step_status(step_name, "错误")
            return False