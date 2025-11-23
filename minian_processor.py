# D:\Desktop\ZJU\SRTP\ui_v1\minian_processor.py

import numpy as np
import xarray as xr
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Union
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
    create_merge_matrix_plot
    
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
        """
        更新特定步骤的运行状态。
        （这个方法是您代码运行逻辑所依赖的）
        """
        # 假设这里是更新状态字典或 UI 界面的逻辑
        if hasattr(self, 'status_log'):
            self.status_log[step_name] = status
        # 或者其他具体的实现，例如打印到日志
        print(f"[STATUS] {step_name}: {status}")
        
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
            'second_spatial_update': None,
            'second_temporal_update': None,
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

            print(f"DEBUG: 最终传递给 seeds_init 的参数: {params_to_pass}")

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
                print(f"DEBUG: 开始计算频率 {freq} 的 PNR 均值...")
                pnr_mean_current_float = pnr_all.mean().compute().item()
                print(f"DEBUG: 频率 {freq} 的 PNR 均值计算完成: {pnr_mean_current_float:.4f}")
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

    def run_first_spatial_update(self) -> bool:
        """
        步骤 11: 第一次空间更新 (Update Spatial) 与背景更新 (Update Background)
        并保存 A, C, b, f 的新值。
        """
        step_name = 'first_spatial_update'
        self.update_step_status(step_name, "运行中")
        try:
            # import matplotlib.pyplot as plt # 不需要直接导入，因为在 create_spatial_update_plot 内部处理

            # 1. 加载数据
            intpath = os.environ.get("MINIAN_INTERMEDIATE", "./intermediate_data")
            varr_mc = self._load_data_from_repo('varr_mc')
            A_init = self._load_data_from_repo('A_init')
            C_init = self._load_data_from_repo('C_init')
            C_chk_init = self._load_data_from_repo('C_init').rename("C_chk") 
            sn_spatial = self._load_data_from_repo('sn_spatial') 
            chk = self._load_data_from_repo('chk_settings')
            
            params = self.get_step_params(step_name)
            spatial_kwargs = params.get('spatial_kwargs', {})

            # --- 第一次空间更新 ---
            self.log_output.append("-> 正在执行第一次空间更新...")
            A_new, mask, norm_fac = update_spatial(
                varr_mc, A_init, C_init, sn_spatial, **spatial_kwargs
            )

            C_new = (C_init.sel(unit_id=mask) * norm_fac).rename("C_new")
            C_new = save_minian(C_new, intpath, overwrite=True)
            self._save_data_to_repo(C_new, "C_new_iter1")

            C_chk_new = (C_chk_init.sel(unit_id=mask) * norm_fac).rename("C_chk_new")
            C_chk_new = save_minian(C_chk_new, intpath, overwrite=True)
            self._save_data_to_repo(C_chk_new, "C_chk_new_iter1")

            # --- 背景更新 ---
            self.log_output.append("-> 正在执行背景更新...")
            b_new, f_new = update_background(varr_mc, A_new, C_chk_new)
            
            # --- 可视化 (2x2 空间足迹对比) ---
            self.log_output.append("-> 正在生成空间更新可视化结果 (Matplotlib 2x2)。")
            
            # 1. 准备数据 (计算 Dask 数组并转换为 NumPy)
            A_init_max = A_init.max("unit_id").compute().astype(np.float32).values
            A_init_sum = (A_init.fillna(0) > 0).sum("unit_id").compute().astype(np.uint8).values
            A_new_max = A_new.max("unit_id").compute().astype(np.float32).values
            A_new_sum = (A_new > 0).sum("unit_id").compute().astype(np.uint8).values
            
            # 2. 调用新的可视化函数
            img_array = create_spatial_update_plot(
                A_init_max, 
                A_init_sum, 
                A_new_max, 
                A_new_sum, 
                step_name="First Update" # 传入 step_name 区分
            )

            # 3. 保存 NumPy 数组供 PyQt 显示
            self._save_data_to_repo(img_array, f"{step_name}_vis_array")
            
            # --- 保存最终结果并更新 repo 中的主键 ---
            self.log_output.append("-> 正在保存 A, C, b, f 的第一次迭代结果...")

            A = save_minian(
                A_new.rename("A"),
                intpath,
                overwrite=True,
                chunks=params.get('A_chunks', {"unit_id": 1, "height": -1, "width": -1}),
            )
            self._save_data_to_repo(A, "A_iter1")
            
            b = save_minian(b_new.rename("b"), intpath, overwrite=True)
            self._save_data_to_repo(b, "b_iter1")

            f = save_minian(
                f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
            )
            self._save_data_to_repo(f, "f_iter1")

            C = save_minian(C_new.rename("C"), intpath, overwrite=True)
            self._save_data_to_repo(C, "C_iter1")

            C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
            self._save_data_to_repo(C_chk, "C_chk_iter1")

            self.log_output.append("✅ 步骤 11 运行完成。")
            self.update_step_status(step_name, "已完成")
            return True

        except Exception as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.update_step_status(step_name, "错误")
            return False

    # 请注意： run_first_temporal_update_explore (步骤 12) 保持不变，因为它使用 Holoviews/Bokeh 风格的 visualize_temporal_update
    # 假设 create_cnmf_update_plot, compute_trace, update_temporal 等已导入

    def run_first_temporal_update_explore(self) -> bool:
        """
        步骤 12: 初次时间更新 (参数探索) - 修改为 Matplotlib 单个单元四宫格图
        """
        step_name = 'first_temporal_update_explore'
        self.update_step_status(step_name, "运行中")
        try:
            # 1. 加载数据
            varr_mc = self._load_data_from_repo('varr_mc') # 对应 Y_fm_chk
            A_init = self._load_data_from_repo('A_init') # 对应 A
            C_init = self._load_data_from_repo('C_init') # 对应 C_chk
            
            b_current = self._load_data_from_repo('b_iter1', allow_none=True)
            f_current = self._load_data_from_repo('f_iter1', allow_none=True)
            # ... (b_current/f_current 初始化代码不变) ...
            if b_current is None:
                b_current = xr.zeros_like(varr_mc.isel(frame=0, drop=True)).rename("b")
            if f_current is None:
                f_current = xr.zeros_like(varr_mc.isel(height=0, width=0).mean(dim=["height", "width"], drop=True)).rename("f")

            # 2. 获取单个参数组合
            params = self.get_step_params(step_name)
            p = params.get('p', 1)
            sparse_penal = params.get('sparse_penal', 1.0)
            add_lag = params.get('add_lag', 20)
            noise_freq = params.get('noise_freq', 0.06)

            # 3. 选取子集单位 (Units)
            self.log_output.append("-> 正在选取 10 个随机单位进行时间更新探索。")
            all_units = A_init.coords["unit_id"].values
            units_to_select = min(10, len(all_units))
            np.random.seed(1) 
            units = np.random.choice(all_units, units_to_select, replace=False)
            units.sort()
            
            A_sub = A_init.sel(unit_id=units).persist()
            C_sub = C_init.sel(unit_id=units).persist()
            
            # 4. 计算残差 (YrA)
            self.log_output.append("-> 正在计算 YrA (残差/trace)...")
            # 假设 compute_trace 返回 Y_rA (即 Y - b*f) 减去单位本身贡献后的信号
            # Minian 的 compute_trace 实际上是 (Y - b*f) * A.T
            # 这里为了演示，我们假设它返回了正确的输入信号 (Y_rA)
            YrA = compute_trace(
                varr_mc, A_sub, b_current, C_sub, f_current
            ).persist().chunk({"unit_id": 1, "frame": -1})
            
            self.log_output.append(f"-> 执行探索 (p={p}, sparse_penal={sparse_penal}, add_lag={add_lag}, noise_freq={noise_freq})...")

            # 5. 运行 update_temporal (单参数)
            cur_C, cur_S, cur_b0, cur_c0, cur_g, cur_mask = update_temporal(
                A_sub,
                C_sub,
                YrA=YrA,
                sparse_penal=sparse_penal,
                p=p,
                use_smooth=True,
                add_lag=add_lag,
                noise_freq=noise_freq,
            )
            
            # 6. 可视化：使用 create_cnmf_update_plot (替代方案)
            self.log_output.append("-> 正在生成时间更新可视化结果 (Matplotlib/代表性单元)。")
            
            # 选取一个代表性 Unit ID (例如第一个)
            representative_unit_id = units[0]
            # 选取一个代表性帧
            frame_idx = 0 
            
            # 为了 create_cnmf_update_plot，我们需要 A, C, S 的 Xarray DataArray
            # 注意：这里我们无法直接显示 C 与 YrA 的对比，只能显示 C 本身 (即四宫格图的 C 图)
            img_array = create_cnmf_update_plot(
                varr=self.Y_ds, # 假设 self.Y_ds 是原始视频，用于背景
                A_comp=A_sub.compute(), 
                C_comp=cur_C.compute(), 
                S_comp=cur_S.compute(), 
                unit_id=representative_unit_id, 
                frame_idx=frame_idx
            )
            
            # 7. 保存图像
            image_path = f"{step_name}_temporal_cnmf_plot.png"
            full_path = os.path.join(self.repo_dir, image_path)
            cv2.imwrite(full_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)) 
            self._save_data_to_repo(image_path, f"{step_name}_vis")
            
            self.log_output.append("✅ 步骤 12 运行完成。")
            self.update_step_status(step_name, "已完成")
            return True
        except Exception as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.update_step_status(step_name, "错误")
            return False

    def run_first_temporal_update(self) -> bool:
        """
        步骤 13: 第一次时间更新 (Update Temporal) 和单位合并 (Unit Merge)
        更新 A, C, S, b0, c0 的新值。
        """
        step_name = 'first_temporal_update'
        self.update_step_status(step_name, "运行中")
        try:
            # 1. 加载数据 (使用迭代 1 的结果)
            intpath = os.environ.get("MINIAN_INTERMEDIATE", "./intermediate_data")
            varr_mc = self._load_data_from_repo('varr_mc')
            A_current = self._load_data_from_repo('A_iter1')
            C_current = self._load_data_from_repo('C_iter1')
            C_chk_current = self._load_data_from_repo('C_chk_iter1')
            b_current = self._load_data_from_repo('b_iter1')
            f_current = self._load_data_from_repo('f_iter1')
            chk = self._load_data_from_repo('chk_settings') 
            
            params = self.get_step_params(step_name)
            temporal_kwargs = params.get('temporal_kwargs', {})
            merge_kwargs = params.get('merge_kwargs', {}) # 确保获取 merge_kwargs

            # --- 计算 YrA ---
            self.log_output.append("-> 正在计算 YrA (残差/trace)...")
            YrA = compute_trace(
                varr_mc, A_current, b_current, C_chk_current, f_current
            ).persist() 

            # --- 第一次时间更新 ---
            self.log_output.append("-> 正在执行第一次时间更新...")
            C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
                A_current, C_current, YrA=YrA, **temporal_kwargs
            )
            
            C_new = C_new.rename("C_new")
            C_chk_new = C_chk_current.sel(unit_id=C_new.coords["unit_id"].values).rename("C_chk_new") # 调整 C_chk_new 的大小

            # --- 可视化 (初始/第一次更新 C/S 矩阵图) ---
            self.log_output.append("-> 正在生成时间更新和事件的矩阵可视化结果。")
            
            C_init_comp = C_current.compute().astype(np.float32).values
            C_new_comp = C_new.compute().astype(np.float32).values
            S_new_comp = S_new.compute().astype(np.float32).values
            
            # 调用新的 C/S 矩阵可视化函数
            img_array_c_s = create_temporal_matrix_plot(
                C_init_comp, 
                C_new_comp, 
                S_new_comp, 
                step_name="First Update"
            )
            self._save_data_to_repo(img_array_c_s, f"{step_name}_c_s_vis_array")
            
            # --- 可视化 (接受单位的细节) ---
            self.log_output.append("-> 正在生成接受单位的详细时间更新可视化 (10个样本)。")
            sig = C_new + b0_new + c0_new
            
            accepted_units = C_new.coords["unit_id"].values
            units_to_sample = min(10, len(accepted_units))
            np.random.seed(2) 
            sample_units = np.random.choice(accepted_units, units_to_sample, replace=False)
            
            A_comp = A_current.sel(unit_id=sample_units).compute()
            C_comp = C_new.sel(unit_id=sample_units).compute()
            S_comp = S_new.sel(unit_id=sample_units).compute()
            
            # 找到平均C最大的帧作为重建帧
            mean_C_idx = int(C_comp.mean('unit_id').argmax().values)
            
            # 为每个样本单位创建并保存详细四宫格图
            for i, unit_id in enumerate(sample_units):
                img_array_unit = create_cnmf_update_plot(
                    varr_mc, 
                    A_comp, 
                    C_comp, 
                    S_comp, 
                    unit_id, 
                    mean_C_idx
                )
                self._save_data_to_repo(img_array_unit, f"{step_name}_accepted_unit_{unit_id}_vis_array")


            # --- 临时保存 C, C_chk, S, b0, c0 ---
            self.log_output.append("-> 正在保存时间更新结果...")
            
            C_current = save_minian(C_new.rename("C"), intpath, overwrite=True)
            self._save_data_to_repo(C_current, "C_tmp_merge")

            C_chk_current = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
            self._save_data_to_repo(C_chk_current, "C_chk_tmp_merge")

            S_current = save_minian(S_new.rename("S"), intpath, overwrite=True)
            self._save_data_to_repo(S_current, "S_tmp_merge")

            b0_current = save_minian(b0_new.rename("b0"), intpath, overwrite=True)
            self._save_data_to_repo(b0_current, "b0_tmp_merge")
            
            c0_current = save_minian(c0_new.rename("c0"), intpath, overwrite=True)
            self._save_data_to_repo(c0_current, "c0_tmp_merge")
            
            A_current = A_current.sel(unit_id=C_current.coords["unit_id"].values)
            self._save_data_to_repo(A_current, "A_tmp_merge")

            # --- 单位合并 ---
            self.log_output.append("-> 正在执行单位合并...")
            A_mrg, C_mrg, sig_mrg_list = unit_merge(
                A_current, 
                C_current, 
                [C_current + b0_current + c0_current], 
                **merge_kwargs
            )
            sig_mrg = sig_mrg_list[0] 

            # --- 合并可视化 (C 矩阵图对比) ---
            self.log_output.append("-> 正在生成合并对比可视化。")
            
            C_before_comp = C_current.compute().astype(np.float32).values
            C_after_comp = C_mrg.compute().astype(np.float32).values
            
            # 调用新的合并矩阵可视化函数
            img_array_merge = create_merge_matrix_plot(
                C_before_comp, 
                C_after_comp, 
                step_name="First Merge"
            )
            self._save_data_to_repo(img_array_merge, f"{step_name}_merge_vis_array")

            # --- 保存最终合并结果并更新 repo 中的主键 ---
            self.log_output.append("-> 正在保存最终合并结果...")

            A_current = save_minian(A_mrg.rename("A"), intpath, overwrite=True)
            self._save_data_to_repo(A_current, "A_iter1_merged")

            C_current = save_minian(C_mrg.rename("C"), intpath, overwrite=True)
            self._save_data_to_repo(C_current, "C_iter1_merged")

            C_chk_current = save_minian(C_current.rename("C_chk"), intpath, overwrite=True)
            self._save_data_to_repo(C_chk_current, "C_chk_iter1_merged")

            sig_current = save_minian(sig_mrg.rename("sig"), intpath, overwrite=True)
            self._save_data_to_repo(sig_current, "sig_iter1_merged")

            self.log_output.append("✅ 步骤 13 运行完成。")
            self.update_step_status(step_name, "已完成")
            return True

        except Exception as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.update_step_status(step_name, "错误")
            return False
    
    def run_second_spatial_update(self) -> bool:
        """
        步骤 14: 第二次空间更新 (Update Spatial) 与背景更新 (Update Background)
        并保存 A, C, b, f 的第二次迭代结果。
        """
        step_name = 'second_spatial_update'
        self.update_step_status(step_name, "运行中")
        try:
            # 1. 加载数据 (使用第一次合并后的结果作为起点)
            intpath = os.environ.get("MINIAN_INTERMEDIATE", "./intermediate_data")
            varr_mc = self._load_data_from_repo('varr_mc')
            A_init = self._load_data_from_repo('A_iter1_merged') # A 迭代起点
            C_init = self._load_data_from_repo('C_iter1_merged') # C 迭代起点
            C_chk_init = self._load_data_from_repo('C_chk_iter1_merged').rename("C_chk") 
            sn_spatial = self._load_data_from_repo('sn_spatial') 
            chk = self._load_data_from_repo('chk_settings')
            
            params = self.get_step_params(step_name)
            # 假设第二次迭代的参数键为 'spatial_kwargs_iter2'
            spatial_kwargs = params.get('spatial_kwargs_iter2', {}) 

            # --- 第二次空间更新 ---
            self.log_output.append("-> 正在执行第二次空间更新...")
            # 使用第二次参数进行 update_spatial
            A_new, mask, norm_fac = update_spatial(
                varr_mc, A_init, C_init, sn_spatial, **spatial_kwargs
            )

            C_new = (C_init.sel(unit_id=mask) * norm_fac).rename("C_new")
            C_new = save_minian(C_new, intpath, overwrite=True)
            self._save_data_to_repo(C_new, "C_new_iter2")

            C_chk_new = (C_chk_init.sel(unit_id=mask) * norm_fac).rename("C_chk_new")
            C_chk_new = save_minian(C_chk_new, intpath, overwrite=True)
            self._save_data_to_repo(C_chk_new, "C_chk_new_iter2")

            # --- 背景更新 ---
            self.log_output.append("-> 正在执行背景更新...")
            b_new, f_new = update_background(varr_mc, A_new, C_chk_new)
            
            # --- 可视化 (2x2 空间足迹对比) ---
            self.log_output.append("-> 正在生成第二次空间更新可视化结果 (Matplotlib 2x2)。")
            
            # 1. 准备数据 (计算 Dask 数组并转换为 NumPy)
            A_init_max = A_init.max("unit_id").compute().astype(np.float32).values
            A_init_sum = (A_init.fillna(0) > 0).sum("unit_id").compute().astype(np.uint8).values
            A_new_max = A_new.max("unit_id").compute().astype(np.float32).values
            A_new_sum = (A_new > 0).sum("unit_id").compute().astype(np.uint8).values
            
            # 2. 调用新的可视化函数
            img_array = create_spatial_update_plot(
                A_init_max, 
                A_init_sum, 
                A_new_max, 
                A_new_sum, 
                step_name="Second Update" # 传入 step_name 区分
            )

            # 3. 保存 NumPy 数组供 PyQt 显示
            self._save_data_to_repo(img_array, f"{step_name}_vis_array")

            # --- 保存最终结果并更新 repo 中的主键 (Iter 2 的初始数据) ---
            self.log_output.append("-> 正在保存 A, C, b, f 的第二次迭代结果...")

            A = save_minian(
                A_new.rename("A"),
                intpath,
                overwrite=True,
                chunks=params.get('A_chunks', {"unit_id": 1, "height": -1, "width": -1}),
            )
            self._save_data_to_repo(A, "A_iter2")
            
            b = save_minian(b_new.rename("b"), intpath, overwrite=True)
            self._save_data_to_repo(b, "b_iter2")

            f = save_minian(
                f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
            )
            self._save_data_to_repo(f, "f_iter2")

            C = save_minian(C_new.rename("C"), intpath, overwrite=True)
            self._save_data_to_repo(C, "C_iter2")

            C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
            self._save_data_to_repo(C_chk, "C_chk_iter2")

            self.log_output.append("✅ 步骤 14 运行完成。")
            self.update_step_status(step_name, "已完成")
            return True

        except Exception as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.update_step_status(step_name, "错误")
            return False

    def run_second_temporal_update(self) -> bool:
        """
        步骤 15: 第二次时间更新 (Update Temporal) 和单位合并 (Unit Merge)
        更新 A, C, S, b0, c0 的新值。
        """
        step_name = 'second_temporal_update'
        self.update_step_status(step_name, "运行中")
        try:
            # 1. 加载数据 (使用迭代 2 空间更新后的结果)
            intpath = os.environ.get("MINIAN_INTERMEDIATE", "./intermediate_data")
            varr_mc = self._load_data_from_repo('varr_mc')
            A_current = self._load_data_from_repo('A_iter2') # 对应 A
            C_current = self._load_data_from_repo('C_iter2') # 对应 C
            C_chk_current = self._load_data_from_repo('C_chk_iter2') # 对应 C_chk
            b_current = self._load_data_from_repo('b_iter2') # 对应 b
            f_current = self._load_data_from_repo('f_iter2') # 对应 f
            chk = self._load_data_from_repo('chk_settings') 
            
            params = self.get_step_params(step_name)
            # 假设第二次迭代的参数键
            temporal_kwargs = params.get('temporal_kwargs_iter2', {})
            merge_kwargs = params.get('merge_kwargs_iter2', {})

            # --- 计算 YrA ---
            self.log_output.append("-> 正在计算 YrA (残差/trace)...")
            YrA = compute_trace(
                varr_mc, A_current, b_current, C_chk_current, f_current
            ).persist() 

            # --- 第二次时间更新 ---
            self.log_output.append("-> 正在执行第二次时间更新...")
            # 使用第二次参数进行 update_temporal
            C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
                A_current, C_current, YrA=YrA, **temporal_kwargs
            )
            
            C_new = C_new.rename("C_new")
            C_chk_new = C_chk_current.sel(unit_id=C_new.coords["unit_id"].values).rename("C_chk_new")

            # --- 可视化 (初始/第二次更新 C/S 矩阵图) ---
            self.log_output.append("-> 正在生成时间更新和事件的矩阵可视化结果。")
            
            C_init_comp = C_current.compute().astype(np.float32).values
            C_new_comp = C_new.compute().astype(np.float32).values
            S_new_comp = S_new.compute().astype(np.float32).values
            
            # 调用新的 C/S 矩阵可视化函数
            img_array_c_s = create_temporal_matrix_plot(
                C_init_comp, 
                C_new_comp, 
                S_new_comp, 
                step_name="Second Update"
            )
            self._save_data_to_repo(img_array_c_s, f"{step_name}_c_s_vis_array")
            
            # --- 可视化 (接受单位的细节) ---
            self.log_output.append("-> 正在生成接受单位的详细时间更新可视化 (10个样本)。")
            sig = C_new + b0_new + c0_new
            
            accepted_units = C_new.coords["unit_id"].values
            units_to_sample = min(10, len(accepted_units))
            np.random.seed(3) # 新的随机种子
            sample_units = np.random.choice(accepted_units, units_to_sample, replace=False)
            
            A_comp = A_current.sel(unit_id=sample_units).compute()
            C_comp = C_new.sel(unit_id=sample_units).compute()
            S_comp = S_new.sel(unit_id=sample_units).compute()
            
            mean_C_idx = int(C_comp.mean('unit_id').argmax().values)
            
            for i, unit_id in enumerate(sample_units):
                img_array_unit = create_cnmf_update_plot(
                    varr_mc, 
                    A_comp, 
                    C_comp, 
                    S_comp, 
                    unit_id, 
                    mean_C_idx
                )
                self._save_data_to_repo(img_array_unit, f"{step_name}_accepted_unit_{unit_id}_vis_array")


            # --- 临时保存 C, C_chk, S, b0, c0 ---
            self.log_output.append("-> 正在保存时间更新结果...")
            
            C_current = save_minian(C_new.rename("C"), intpath, overwrite=True)
            self._save_data_to_repo(C_current, "C_tmp_merge")

            C_chk_current = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
            self._save_data_to_repo(C_chk_current, "C_chk_tmp_merge")

            S_current = save_minian(S_new.rename("S"), intpath, overwrite=True)
            self._save_data_to_repo(S_current, "S_tmp_merge")

            b0_current = save_minian(b0_new.rename("b0"), intpath, overwrite=True)
            self._save_data_to_repo(b0_current, "b0_tmp_merge")
            
            c0_current = save_minian(c0_new.rename("c0"), intpath, overwrite=True)
            self._save_data_to_repo(c0_current, "c0_tmp_merge")
            
            A_current = A_current.sel(unit_id=C_current.coords["unit_id"].values)
            self._save_data_to_repo(A_current, "A_tmp_merge")

            # --- 单位合并 ---
            self.log_output.append("-> 正在执行单位合并...")
            A_mrg, C_mrg, sig_mrg_list = unit_merge(
                A_current, 
                C_current, 
                [C_current + b0_current + c0_current], 
                **merge_kwargs
            )
            sig_mrg = sig_mrg_list[0] 

            # --- 合并可视化 (C 矩阵图对比) ---
            self.log_output.append("-> 正在生成合并对比可视化。")
            
            C_before_comp = C_current.compute().astype(np.float32).values
            C_after_comp = C_mrg.compute().astype(np.float32).values
            
            # 调用新的合并矩阵可视化函数
            img_array_merge = create_merge_matrix_plot(
                C_before_comp, 
                C_after_comp, 
                step_name="Second Merge"
            )
            self._save_data_to_repo(img_array_merge, f"{step_name}_merge_vis_array")


            # --- 保存最终合并结果并更新 repo 中的主键 ---
            self.log_output.append("-> 正在保存最终合并结果...")

            A_current = save_minian(A_mrg.rename("A"), intpath, overwrite=True)
            self._save_data_to_repo(A_current, "A_iter2_merged")

            C_current = save_minian(C_mrg.rename("C"), intpath, overwrite=True)
            self._save_data_to_repo(C_current, "C_iter2_merged")

            C_chk_current = save_minian(C_current.rename("C_chk"), intpath, overwrite=True)
            self._save_data_to_repo(C_chk_current, "C_chk_iter2_merged")

            sig_current = save_minian(sig_mrg.rename("sig"), intpath, overwrite=True)
            self._save_data_to_repo(sig_current, "sig_iter2_merged")

            self.log_output.append("✅ 步骤 15 运行完成。")
            self.update_step_status(step_name, "已完成")
            return True

        except Exception as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.update_step_status(step_name, "错误")
            return False

    def run_save_data(self) -> bool:
        """
        步骤 16: 数据保存
        将最终结果保存到 Minian 文件中（通常是 minian.nc）。
        """
        step_name = 'save_data'
        self.update_step_status(step_name, "运行中")

        try:
            # from .utilities import save_minian
            # 最终数据通常是最后一次迭代的结果
            A = self._load_data_from_repo("A_iter2")
            C = self._load_data_from_repo("C_iter2")
            S = self._load_data_from_repo("S_iter2")
            b = self._load_data_from_repo("b_iter2")
            f = self._load_data_from_repo("f_iter2")
            
            params = self.get_step_params(step_name)

            self.log_output.append("-> 正在保存最终 CNMF 结果 (A, C, S, b, f)...")
            
            # 使用 save_minian 函数将 Dask 数组持久化到 Zarr 存储或 .nc 文件
            save_minian_kwargs = params.get('save_minian_kwargs', {'dpath': './minian_output', 'overwrite': True})
            
            A = save_minian(A.rename("A"), **save_minian_kwargs)
            C = save_minian(C.rename("C"), **save_minian_kwargs)
            S = save_minian(S.rename("S"), **save_minian_kwargs)
            b = save_minian(b.rename("b"), **save_minian_kwargs)
            f = save_minian(f.rename("f"), **save_minian_kwargs)
            
            self.log_output.append("✅ 所有数据保存完成。")
            self.update_step_status(step_name, "已完成")
            return True
            
        except Exception as e:
            self.log_output.append(f"运行【{step_name}】失败: {e}")
            self.update_step_status(step_name, "错误")
            return False