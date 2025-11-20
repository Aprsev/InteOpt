import functools as fct
import itertools as itt
import os
from typing import List, Optional, Tuple, Union

import cv2
import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage.measurements import center_of_mass

# 导入本地 minian 核心工具
from .cnmf import compute_AtC
from .motion_correction import apply_shifts
from .utilities import rechunk_like, save_minian

# =========================================================================
# 核心辅助函数
# =========================================================================

def normalize_frame(frame: np.ndarray, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """将帧数据归一化到 0-255 范围，返回 np.uint8 灰度图。"""
    frame = frame.astype(np.float32)
    
    if vmin is None:
        vmin = frame.min()
    if vmax is None:
        vmax = frame.max()

    range_val = vmax - vmin
    if range_val <= 1e-6:
        # 避免除以零或几乎为零
        norm_frame = np.zeros_like(frame)
    else:
        norm_frame = (frame - vmin) / range_val * 255.0
        
    return np.clip(norm_frame, 0, 255).astype(np.uint8)


def get_single_frame_vis(varr: xr.DataArray, frame_idx: int) -> np.ndarray:
    """
    提取并返回单帧作为 NumPy 数组 (H, W) 的浮点数据。
    这是所有视频类可视化（步骤 1-5）的基础。

    参数
    ----------
    varr : xr.DataArray
        输入的视频数据。
    frame_idx : int
        要提取的帧索引。

    返回
    -------
    np.ndarray
        单帧数据 (H, W)，浮点类型。
    """
    try:
        # 使用 compute() 强制计算 Dask 数组
        # 注意：如果 varr 很大，这可能会阻塞
        frame = varr.isel(frame=frame_idx).compute().values
        return frame.astype(np.float32)
    except Exception as e:
        print(f"错误: 提取帧 {frame_idx} 失败: {e}")
        # 返回一个全零数组作为占位符
        if varr.ndim == 3:
            h, w = varr.height.size, varr.width.size 
            return np.zeros((h, w), dtype=np.float32)
        else:
             # 对于非视频数据，返回一个 100x100 的占位符
            return np.zeros((100, 100), dtype=np.float32)

def centroid(A: xr.DataArray, verbose: bool = False) -> pd.DataFrame:
    """
    计算空间足迹的质心 (简化版本)。
    """
    if verbose:
        print("正在计算质心...")
    
    A_val = A.fillna(0).compute().values
    centroids_list = []
    
    for uid in range(A_val.shape[0]):
        # Center of mass 返回 (height, width) 坐标 (row, col)
        cy, cx = center_of_mass(A_val[uid, :, :])
        
        # Minian 通常使用 'width' (x) 和 'height' (y)
        centroids_list.append({
            'unit_id': A.unit_id.values[uid] if 'unit_id' in A.coords else uid, 
            'height': cy, 
            'width': cx
        })
        
    return pd.DataFrame(centroids_list)

# =========================================================================
# 步骤可视化函数 (返回 NumPy 图像数组)
# =========================================================================

def get_normalized_video_frame(varr: xr.DataArray, frame_idx: int) -> np.ndarray:
    """
    获取单个归一化的视频帧，用于步骤 1, 2, 3, 4 的单视频显示。
    
    返回 BGR 格式 (H, W, 3) 数组，方便 cv2 显示。
    """
    frame = get_single_frame_vis(varr, frame_idx)
    norm_frame = normalize_frame(frame)
    # 转换为 3 通道 BGR 图像 (OpenCV 格式)
    return cv2.cvtColor(norm_frame, cv2.COLOR_GRAY2BGR)


def create_mc_max_projection_comparison(varr_in: xr.DataArray, varr_mc: xr.DataArray) -> np.ndarray:
    """
    生成运动校正前后最大投影图的对比可视化。

    参数
    ----------
    varr_in : xr.DataArray
        运动校正前的视频数据 (MinianProcessor.varr_in)。
    varr_mc : xr.DataArray
        运动校正后的视频数据 (MinianProcessor.varr_mc)。

    返回
    -------
    np.ndarray
        包含 1x2 对比图的 RGB 图像数组 (H, W*2, 3)。
    """
    print("执行: 生成运动校正前后最大投影图对比...")
    # 确保支持中文标题
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

    # 1. 计算最大投影 (使用 dask 自动计算)
    # 沿时间维度 'frame' 取最大值
    max_proj_in = varr_in.max(dim='frame').compute().values
    max_proj_mc = varr_mc.max(dim='frame').compute().values
    
    # 2. 统一对比度
    vmin = min(max_proj_in.min(), max_proj_mc.min())
    vmax = max(max_proj_in.max(), max_proj_mc.max())

    # 3. 使用 Matplotlib 绘制对比图 (提供清晰的标题和布局)
    # figsize(10, 5) 适应常见的 UI 布局
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=100) 
    
    # Plot 1: Before MC
    axes[0].imshow(max_proj_in, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title("MC 前最大投影")
    axes[0].axis('off')

    # Plot 2: After MC
    axes[1].imshow(max_proj_mc, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title("MC 后最大投影")
    axes[1].axis('off')
    
    plt.tight_layout()

    # 4. 将 Matplotlib 图转换为 NumPy 数组
    fig.canvas.draw()
    img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return img_data


def _convert_seeds_to_df(seeds_data: Union[pd.DataFrame, xr.DataArray]) -> pd.DataFrame:
    """辅助函数：将 XArray 种子转换为规范的 DataFrame。"""
    if isinstance(seeds_data, xr.DataArray):
        # 转换xarray.DataArray为pandas.DataFrame
        try:
            # 尝试使用 initialization.py 中新的坐标结构
            seeds_df = pd.DataFrame({
                'height': seeds_data.coords['height'].values,
                'width': seeds_data.coords['width'].values
            })
        except KeyError:
            # 回退到旧的 DataArray 结构
            seeds_df = seeds_data.rename("seeds").to_dataframe().reset_index()
            if 'height' not in seeds_df.columns or 'width' not in seeds_df.columns:
                if 'dim_0' in seeds_df.columns and 'dim_1' in seeds_df.columns:
                    seeds_df['height'] = seeds_df['dim_0']
                    seeds_df['width'] = seeds_df['dim_1']
                else:
                    raise ValueError("输入DataArray必须包含height/width或dim_0/dim_1维度")
        return seeds_df
    else:
        # 假设已经是 DataFrame
        return seeds_data.copy()

def _draw_seeds(vis_frame: np.ndarray, seeds: pd.DataFrame, color: Tuple[int, int, int]):
    """辅助函数：在图像上绘制一组特定颜色的种子。"""
    for idx, row in seeds.iterrows():
        # 确保坐标是整数
        try:
            cx = int(row['width'])
            cy = int(row['height'])
            # 绘制圆圈 (BGR 格式)
            cv2.circle(vis_frame, (cx, cy), 1, color, -1) 
        except (KeyError, ValueError):
            # 忽略无效的行
            pass

def create_seeds_visualization(
    varr_max_proj: xr.DataArray, 
    seeds_kept: Union[pd.DataFrame, xr.DataArray], 
    seeds_removed: Optional[Union[pd.DataFrame, xr.DataArray]] = None
) -> np.ndarray:
    """
    在最大投影图像上叠加种子点。
    
    - seeds_kept (必需): 绘制为白色的种子 (DataFrame 或 xr.DataArray)
    - seeds_removed (可选): 绘制为红色的种子 (DataFrame 或 xr.DataArray)
    
    返回叠加了种子点的 BGR 图像 (H, W, 3) 数组。
    """
    print(f"执行: 生成叠加种子点 (基于最大投影)...")
    
    # 定义颜色 (BGR 格式)
    COLOR_KEPT = (255, 255, 255) # 白色
    COLOR_REMOVED = (0, 0, 255)   # 红色

    # 1. 使用最大投影作为背景
    frame = varr_max_proj.values # 确保是 numpy 数组
    
    # 2. 归一化帧并转换为 3 通道 BGR 图像
    norm_frame = normalize_frame(frame)
    vis_frame = cv2.cvtColor(norm_frame, cv2.COLOR_GRAY2BGR) 

    # 3. 处理并绘制 "移除" 的种子 (红色)
    # (先绘制红色，这样白色可以覆盖在上面，以防万一有重叠)
    if seeds_removed is not None:
        try:
            current_seeds_removed = _convert_seeds_to_df(seeds_removed)
            print(f"-> 正在绘制 {len(current_seeds_removed)} 个 '移除' 的种子 (红色)...")
            _draw_seeds(vis_frame, current_seeds_removed, COLOR_REMOVED)
        except Exception as e:
            print(f"警告: 无法处理 '移除' 的种子: {e}")

    # 4. 处理并绘制 "保留" 的种子 (白色)
    try:
        current_seeds_kept = _convert_seeds_to_df(seeds_kept)
        print(f"-> 正在绘制 {len(current_seeds_kept)} 个 '保留' 的种子 (白色)...")
        _draw_seeds(vis_frame, current_seeds_kept, COLOR_KEPT)
    except Exception as e:
        print(f"错误: 无法处理 '保留' 的种子: {e}")
        raise # 保留的种子是必需的
        
    return vis_frame

def create_pnr_refine_plot(signals_arr, noises_arr, freq_list, sample_seeds, fs=30.0) -> np.ndarray:
    """
    根据 run_noise_freq_exploration 输出绘制信号与噪声曲线对比图。
    """
    n_freq, n_samples, n_frames = signals_arr.shape
    fig, axes = plt.subplots(2, 3, figsize=(4*2, 1.5*3), squeeze=False)

    t = np.arange(n_frames) / fs
    
    print("The shape of signals_arr and noises_arr is:")
    print(signals_arr.shape, noises_arr.shape)

    for i, freq in enumerate(freq_list):
        for k  in range(2):
            for j in range(3):
                ax = axes[k, j]
                ax.plot(t, signals_arr[i, (k+1)*(j+1)-1], 'r-', label="信号", alpha=0.7)
                ax.plot(t, noises_arr[i, (k+1)*(j+1)-1], 'b-', label="噪声", alpha=0.7)
                ax.set_title(f"Freq={freq:.3f}, Seed#{(k+1)*(j+1)-1}")
                ax.set_xlabel("时间 (s)")
                ax.set_ylabel("幅值")
                if i == 0 and j == 0:
                    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img_array

def create_exploration_plot(
    varr: xr.DataArray, 
    A_list: List[xr.DataArray], 
    penalties: List[float], 
    frame_idx: int = 0
) -> np.ndarray:
    """
    创建稀疏度惩罚探索的可视化（多图），用于步骤 11, 13。

    返回包含多子图的 RGB 图像数组。
    """
    print("执行: 生成参数探索可视化图...")
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    
    n_plots = len(A_list)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), dpi=100)
    axes = np.ravel(axes) 

    # 获取背景帧 (用作平均投影)
    bg_frame = varr.mean('frame').compute().values 
    bg_norm = normalize_frame(bg_frame)
    
    for i in range(n_plots):
        ax = axes[i]
        # 对 A 求和，得到总的空间组分
        A = A_list[i].sum(dim='unit_id') if A_list[i].ndim == 3 and 'unit_id' in A_list[i].coords else A_list[i].squeeze()
        A_img = A.compute().values

        # 归一化 A，并将其叠加到背景上
        A_norm = (A_img - A_img.min()) / (A_img.max() - A_img.min() + 1e-6)
        
        ax.imshow(bg_norm, cmap='gray')
        ax.imshow(A_norm, cmap='viridis', alpha=0.6)

        ax.set_title(f"惩罚值: {penalties[i]:.2e}")
        ax.axis('off')

    # 隐藏多余的子图
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    
    # 转换为 NumPy 数组
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img_array = img_array.reshape(h, w, 3)
    plt.close(fig)
    return img_array

def create_cnmf_update_plot(
    varr: xr.DataArray, 
    A_comp: xr.DataArray, 
    C_comp: xr.DataArray, 
    S_comp: xr.DataArray, 
    unit_id: int, 
    frame_idx: int
) -> np.ndarray:
    """
    创建 CNMF 更新后的四宫格对比图，用于步骤 12, 14, 15, 16。

    返回包含四宫格图的 RGB 图像数组 (H, W, 3)。
    """
    print(f"执行: 生成 CNMF 单元 {unit_id} 的四宫格图...")
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    
    # 提取所需数据
    A_unit = A_comp.sel(unit_id=unit_id).compute().values
    C_unit = C_comp.sel(unit_id=unit_id).compute().values
    S_unit = S_comp.sel(unit_id=unit_id).compute().values
    
    # 原始帧 (用于背景)
    Y_frame = get_single_frame_vis(varr, frame_idx)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=100)
    
    # --- 1. 空间足迹 A ---
    ax = axes[0, 0]
    A_norm = (A_unit - A_unit.min()) / (A_unit.max() - A_unit.min() + 1e-6)
    Y_avg = varr.mean('frame').compute().values
    Y_norm = normalize_frame(Y_avg)
    ax.imshow(Y_norm, cmap='gray')
    ax.imshow(A_norm, cmap='viridis', alpha=0.6)
    ax.set_title(f"空间足迹 A (Unit {unit_id})")
    ax.axis('off')

    # --- 2. 时间活动 C ---
    ax = axes[0, 1]
    # 假设采样率为 30Hz
    fs = 30.0 
    time_vec = np.arange(len(C_unit)) / fs 
    ax.plot(time_vec, C_unit, label='C', color='C0')
    ax.set_title(f"时间活动 C (Unit {unit_id})")
    ax.set_xlabel("时间 (s)")
    
    # --- 3. 事件 S ---
    ax = axes[1, 0]
    ax.plot(time_vec, S_unit, label='S', color='C1')
    ax.set_title(f"事件 S (Unit {unit_id})")
    ax.set_xlabel("时间 (s)")

    # --- 4. 重建帧 A*C[t] ---
    ax = axes[1, 1]
    # 重建帧
    reconst_frame_unit = A_unit * C_unit[frame_idx]
    
    reconst_norm = normalize_frame(reconst_frame_unit)
    ax.imshow(reconst_norm, cmap='gray')
    
    ax.set_title(f"重建帧 A*C[{frame_idx}]")
    ax.axis('off')

    plt.tight_layout()

    # 转换为 NumPy 数组
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img_array = img_array.reshape(h, w, 3)
    plt.close(fig)
    return img_array

def create_spatial_update_plot(
    A_init: np.ndarray, 
    A_init_bin: np.ndarray, 
    A_new: np.ndarray, 
    A_new_bin: np.ndarray,
    step_name: str
) -> np.ndarray:
    """
    创建 CNMF 空间更新的 2x2 对比图：
    1. 初始 A (Max)
    2. 初始 A (Binary Sum)
    3. 更新后 A_new (Max)
    4. 更新后 A_new (Binary Sum)

    参数
    ----------
    A_init, A_init_bin, A_new, A_new_bin: np.ndarray
        预先计算好的空间足迹 NumPy 数组。
    step_name: str
        步骤名称，用于标题区分 (如 "First" 或 "Second")。

    返回
    -------
    np.ndarray
        包含 2x2 图的 RGB 图像数组 (H, W, 3)。
    """
    print(f"执行: 生成 {step_name} 空间更新 2x2 对比图...")
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=100)
    
    titles = [
        f"Spatial Footprints Initial ({step_name})", 
        f"Binary Spatial Footprints Initial ({step_name})",
        f"Spatial Footprints Updated ({step_name})", 
        f"Binary Spatial Footprints Updated ({step_name})"
    ]
    
    # 确定统一的色彩范围
    vmax_max = max(A_init.max(), A_new.max()) if A_init.size > 0 and A_new.size > 0 else 1.0
    vmin_max = min(A_init.min(), A_new.min()) if A_init.size > 0 and A_new.size > 0 else 0.0
    vmax_sum = max(A_init_bin.max(), A_new_bin.max()) if A_init_bin.size > 0 and A_new_bin.size > 0 else 1.0

    plot_data = [A_init, A_init_bin, A_new, A_new_bin]
    plot_ranges = [(vmin_max, vmax_max), (0, vmax_sum), (vmin_max, vmax_max), (0, vmax_sum)]
    cmaps = ['viridis', 'viridis', 'viridis', 'viridis']
    
    for i, ax in enumerate(axes.flat):
        data = plot_data[i]
        vmin, vmax = plot_ranges[i]
        
        # 确保数据不为空
        if data.size == 0:
            ax.set_title(f"{titles[i]} (无数据)")
            ax.axis('off')
            continue
            
        im = ax.imshow(data, cmap=cmaps[i], vmin=vmin, vmax=vmax)
        ax.set_title(titles[i])
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    # 转换为 NumPy 数组 (RGB)
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w_fig, h_fig = fig.canvas.get_width_height()
    img_array = img_array.reshape(h_fig, w_fig, 3)
    plt.close(fig)
    return img_array


def create_temporal_matrix_plot(
    C_init: np.ndarray, 
    C_new: np.ndarray, 
    S_new: np.ndarray,
    step_name: str
) -> np.ndarray:
    """
    创建 CNMF 时间更新的矩阵 2x2 对比图 (其中一格为空)：
    1. 初始 C 矩阵 (C_init)
    2. 更新后 C 矩阵 (C_new)
    3. 更新后 S 矩阵 (S_new)

    参数
    ----------
    C_init, C_new, S_new: np.ndarray
        时间活动和事件的 NumPy 矩阵 (Frame, Unit ID)。
    step_name: str
        步骤名称，用于标题区分 (如 "First" 或 "Second")。

    返回
    -------
    np.ndarray
        包含 2x2 图的 RGB 图像数组 (H, W, 3)。
    """
    print(f"执行: 生成 {step_name} 时间更新 C/S 矩阵对比图...")
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=100)
    
    # 将矩阵转置为 (Unit ID, Frame) 格式以便更好地可视化时间序列
    C_init_T = C_init.T
    C_new_T = C_new.T
    S_new_T = S_new.T
    
    vmax_c = max(C_init.max(), C_new.max()) if C_init.size > 0 and C_new.size > 0 else 1.0
    vmin_c = min(C_init.min(), C_new.min()) if C_init.size > 0 and C_new.size > 0 else 0.0
    vmax_s = S_new.max() if S_new.size > 0 else 1.0
    vmin_s = S_new.min() if S_new.size > 0 else 0.0

    # Plot 1: Temporal Trace Initial
    ax = axes[0, 0]
    im_c = ax.imshow(C_init_T, aspect='auto', cmap='viridis', vmin=vmin_c, vmax=vmax_c)
    ax.set_title(f"Temporal Trace Initial ({step_name})")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Unit ID")
    fig.colorbar(im_c, ax=ax, fraction=0.046, pad=0.04)

    # Plot 2: Temporal Trace New
    ax = axes[0, 1]
    im_c = ax.imshow(C_new_T, aspect='auto', cmap='viridis', vmin=vmin_c, vmax=vmax_c)
    ax.set_title(f"Temporal Trace Updated ({step_name})")
    ax.set_xlabel("Frame")
    fig.colorbar(im_c, ax=ax, fraction=0.046, pad=0.04)

    # Plot 3: Spikes New
    ax = axes[1, 0]
    im_s = ax.imshow(S_new_T, aspect='auto', cmap='magma', vmin=vmin_s, vmax=vmax_s)
    ax.set_title(f"Spikes Updated ({step_name})")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Unit ID")
    fig.colorbar(im_s, ax=ax, fraction=0.046, pad=0.04)

    # Hide Plot 4
    fig.delaxes(axes[1, 1])

    plt.tight_layout()
    
    # 转换为 NumPy 数组 (RGB)
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_height_width()[::-1] + (3,))
    plt.close(fig)
    return img_array


def create_merge_matrix_plot(
    C_before: np.ndarray, 
    C_after: np.ndarray, 
    step_name: str
) -> np.ndarray:
    """
    创建 CNMF 单位合并后的 1x2 矩阵对比图：
    1. 合并前 C 矩阵
    2. 合并后 C 矩阵

    参数
    ----------
    C_before, C_after: np.ndarray
        合并前后的时间活动 NumPy 矩阵 (Frame, Unit ID)。
    step_name: str
        步骤名称，用于标题区分 (如 "First" 或 "Second")。

    返回
    -------
    np.ndarray
        包含 1x2 图的 RGB 图像数组 (H, W, 3)。
    """
    print(f"执行: 生成 {step_name} 单位合并 C 矩阵对比图...")
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), dpi=100)
    
    # 将矩阵转置为 (Unit ID, Frame) 格式以便更好地可视化时间序列
    C_before_T = C_before.T
    C_after_T = C_after.T
    
    vmax_c_mrg = max(C_before.max(), C_after.max()) if C_before.size > 0 and C_after.size > 0 else 1.0
    vmin_c_mrg = min(C_before.min(), C_after.min()) if C_before.size > 0 and C_after.size > 0 else 0.0

    # Plot 1: Before Merge
    ax = axes[0]    
    im_c_before = ax.imshow(C_before_T, aspect='auto', cmap='viridis', vmin=vmin_c_mrg, vmax=vmax_c_mrg)
    ax.set_title(f"Temporal Signals Before Merge ({step_name})")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Unit ID")
    fig.colorbar(im_c_before, ax=ax, fraction=0.046, pad=0.04)

    # Plot 2: After Merge
    ax = axes[1]
    im_c_after = ax.imshow(C_after_T, aspect='auto', cmap='viridis', vmin=vmin_c_mrg, vmax=vmax_c_mrg)
    ax.set_title(f"Temporal Signals After Merge ({step_name})")
    ax.set_xlabel("Frame")
    fig.colorbar(im_c_after, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    # 转换为 NumPy 数组 (RGB)
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_height_width()[::-1] + (3,))
    plt.close(fig)
    return img_array

def create_init_visualization_plot(
    A_init: xr.DataArray, 
    C_init: xr.DataArray, 
    b_init: xr.DataArray, 
    f_init: xr.DataArray
) -> np.ndarray:
    """
    创建 CNMF 初始化 (A, C, b, f) 的 2x2 可视化面板。
    

    返回:
    np.ndarray
        包含 2x2 图的 RGB 图像数组 (H, W, 3)。
    """
    print("执行: 生成 CNMF 初始化 (A, C, b, f) 2x2 可视化...")
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 计算 Dask 数组
    try:
        A_max_proj = A_init.max("unit_id").compute().astype(np.float32).values
        C_matrix = C_init.compute().astype(np.float32).values
        b_spatial = b_init.compute().astype(np.float32).values
        f_temporal = f_init.compute().astype(np.float32).values
    except Exception as e:
        print(f"错误: 计算 Dask 数组失败: {e}")
        # 返回一个错误图像
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        ax.text(0.5, 0.5, f"计算Dask数组失败:\n{e}", ha='center', va='center', color='red')
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_array

    # 2. 创建 2x2 Matplotlib 图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
    
    # --- 图 1: A (空间足迹最大投影) ---
    ax = axes[0, 0]
    im_A = ax.imshow(A_max_proj, cmap='viridis', aspect='auto')
    ax.set_title(f"初始空间足迹 A (Max Proj, {A_init.sizes['unit_id']} 个单位)")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    fig.colorbar(im_A, ax=ax, fraction=0.046, pad=0.04)

    # --- 图 2: C (时间序列矩阵) ---
    ax = axes[0, 1]
    im_C = ax.imshow(C_matrix, cmap='viridis', aspect='auto')
    ax.set_title(f"初始时间序列 C ({C_init.sizes['unit_id']} 个单位)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Unit ID")
    fig.colorbar(im_C, ax=ax, fraction=0.046, pad=0.04)
    
    # --- 图 3: b (背景空间) ---
    ax = axes[1, 0]
    im_b = ax.imshow(b_spatial, cmap='gray', aspect='auto')
    ax.set_title("初始背景空间 b")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    fig.colorbar(im_b, ax=ax, fraction=0.046, pad=0.04)
    
    # --- 图 4: f (背景时间) ---
    ax = axes[1, 1]
    ax.plot(f_temporal)
    ax.set_title("初始背景时间 f")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Intensity")
    ax.grid(True)

    plt.tight_layout()
    
    # 3. 转换为 NumPy 数组 (RGB)
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w_fig, h_fig = fig.canvas.get_width_height()
    img_array = img_array.reshape(h_fig, w_fig, 3)
    plt.close(fig)
    return img_array