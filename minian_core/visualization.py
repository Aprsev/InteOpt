import functools as fct
import itertools as itt
import os
from typing import Any, Dict, List, Optional, Tuple, Union

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

# 全局中文显示支持（不存在的字体会自动回退）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# =========================================================================
# 核心辅助函数
# =========================================================================
def fig_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    """
    将 Matplotlib Figure 转为 RGB NumPy 数组。
    """
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img_array = img_array.reshape(h, w, 3)
    plt.close(fig)
    return img_array

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


def create_save_data_unit_plot(
    A: Optional[xr.DataArray],
    C: Optional[xr.DataArray],
    unit_id: Optional[int],
    excluded_units: Optional[List[int]] = None,
) -> np.ndarray:
    """
    save_data 步骤交互预览：
    - 左图：单个 unit 的空间足迹
    - 右图：单个 unit 的时间曲线
    """
    excluded_units = excluded_units or []

    fig = plt.figure(figsize=(12, 5), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4])
    ax_spatial = fig.add_subplot(gs[0, 0])
    ax_trace = fig.add_subplot(gs[0, 1])

    # 左侧：空间图
    if A is not None and unit_id is not None and "unit_id" in A.coords and int(unit_id) in set(int(u) for u in A.coords["unit_id"].values):
        a_map = A.sel(unit_id=int(unit_id)).compute().astype(np.float32).values
        im = ax_spatial.imshow(a_map, cmap="viridis")
        ax_spatial.set_title(f"Unit {unit_id} 空间分布")
        ax_spatial.set_xlabel("width")
        ax_spatial.set_ylabel("height")
        fig.colorbar(im, ax=ax_spatial, fraction=0.046, pad=0.04)
    else:
        ax_spatial.imshow(np.zeros((32, 32), dtype=np.float32), cmap="gray")
        ax_spatial.set_title("空间分布不可用")
        ax_spatial.axis("off")

    # 右侧：时间曲线
    if C is not None and unit_id is not None and "unit_id" in C.coords and int(unit_id) in set(int(u) for u in C.coords["unit_id"].values):
        c_trace = C.sel(unit_id=int(unit_id)).compute().astype(np.float32).values
        ax_trace.plot(c_trace, lw=1.2, color="tab:blue")
        ax_trace.set_title(f"Unit {unit_id} 时间曲线")
        ax_trace.set_xlabel("frame")
        ax_trace.set_ylabel("signal")
        ax_trace.grid(alpha=0.25)
    else:
        ax_trace.text(0.5, 0.5, "时间曲线不可用", ha="center", va="center", transform=ax_trace.transAxes)
        ax_trace.set_axis_off()

    ex_text = "无" if len(excluded_units) == 0 else ", ".join(str(int(x)) for x in excluded_units[:12])
    if len(excluded_units) > 12:
        ex_text += " ..."
    fig.suptitle(f"save_data 交互预览 | 已排除 unit_id: {ex_text}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig_to_rgb_array(fig)

def inspect_data_structure(name: str, da: Any):
    """侦察并打印数据结构，帮助用户调试"""
    print(f"\n检查数据对象: [{name}]")
    if isinstance(da, xr.DataArray):
        print(f"  - 类型: xarray.DataArray")
        print(f"  - 维度 (dims): {da.dims}")
        print(f"  - 形状 (shape): {da.shape}")
        print(f"  - 坐标 (coords): {list(da.coords.keys())}")
    elif isinstance(da, np.ndarray):
        print(f"  - 类型: numpy.ndarray")
        print(f"  - 形状 (shape): {da.shape}")
    else:
        print(f"  - 类型: {type(da)} (未知)")
        
def create_save_data_dashboard(
    A: Optional[xr.DataArray],
    C: Optional[xr.DataArray],
    unit_id: Optional[int],
    excluded_units: Optional[List[int]] = None,
    dashboard_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    
    excluded_units = excluded_units or []

    def _build_cache(a_da: Optional[xr.DataArray], c_da: Optional[xr.DataArray]) -> Dict[str, Any]:
        cache = {
            "spatial_base": None,
            "unit_id_map": None,
            "traces": {},
            "height": 512,
            "aspect_ratio": 1.0,
            "unit_ids": []
        }

        print("\n" + "!"*20 + " A 矩阵深度侦察 " + "!"*20)
        if a_da is not None:
            # 1. 原始状态检查
            print(f"[DEBUG] A 矩阵原始类型: {type(a_da.data)}")
            
            # 强制转换为 numpy 并检查
            a_data = a_da.compute()
            a_np = np.array(a_data.values, dtype=np.float32)
            
            uids = a_data.coords['unit_id'].values.astype(int)
            num_units, h, w = a_np.shape
            
            # 2. 统计原始数值 (不筛选 > 0)
            raw_max = np.max(a_np)
            raw_min = np.min(a_np)
            raw_mean = np.mean(a_np)
            nonzero_count = np.count_nonzero(a_np)
            
            print(f"[DEBUG] 矩阵形状: {a_np.shape}")
            print(f"[DEBUG] 数值极值: Max={raw_max:.8f}, Min={raw_min:.8f}")
            print(f"[DEBUG] 平均值: {raw_mean:.8f}")
            print(f"[DEBUG] 非零像素总数: {nonzero_count} / {a_np.size}")

            if nonzero_count == 0:
                print("[ERROR] 警报：A 矩阵确实全是 0，请检查上游 A 矩阵更新/合并逻辑。")
                # 即使全是0，也初始化基础结构防止UI崩溃
                cache["spatial_base"] = np.zeros((h, w), dtype=np.uint8)
                cache["unit_id_map"] = np.full((h, w), -1, dtype=np.int32)
                cache["aspect_ratio"] = w / h
                cache["unit_ids"] = uids.tolist()
            else:
                # 3. 只有有值才进行分位数计算
                mip = np.max(a_np, axis=0)
                pos_values = mip[mip > 0]
                p999 = np.percentile(pos_values, 99.9)
                
                print(f"[DEBUG] 99.9% 分位数: {p999:.8f}")
                
                v_max = p999 if p999 > 0 else raw_max
                mip_norm = np.clip(mip / (v_max + 1e-8), 0, 1)
                cache["spatial_base"] = (mip_norm * 255).astype(np.uint8)
                
                # 建立交互 Map
                unit_id_map = np.full((h, w), -1, dtype=np.int32)
                max_vals = np.full((h, w), -1.0, dtype=np.float32)
                for i, uid in enumerate(uids):
                    plane = a_np[i]
                    p_m = np.max(plane)
                    if p_m <= 0: continue
                    mask = plane > (p_m * 0.2)
                    better = mask & (plane > max_vals)
                    unit_id_map[better] = int(uid)
                    max_vals[better] = plane[better]
                
                cache["unit_id_map"] = unit_id_map
                cache["aspect_ratio"] = w / h
                cache["unit_ids"] = uids.tolist()
        else:
            print("[DEBUG] A 矩阵对象为 None")
        print("!"*50 + "\n")

        # C 矩阵处理 (保持不变)
        if c_da is not None:
            c_data = c_da.compute().astype(np.float32)
            c_np = c_data.values
            uids_c = c_data.coords['unit_id'].values.astype(int)
            for i, uid in enumerate(uids_c):
                cache["traces"][int(uid)] = c_np[i]
        return cache

    # --- 获取或创建缓存 ---
    if not isinstance(dashboard_cache, dict) or "unit_id_map" not in dashboard_cache:
        cache = _build_cache(A, C)
    else:
        cache = dashboard_cache
    
    spatial_base = cache["spatial_base"]
    unit_id_map = cache["unit_id_map"]
    traces = cache["traces"]
    target_h = cache.get("height", 512)
    aspect = cache.get("aspect_ratio", 1.0)
    curve_w = 600

    # --- 渲染逻辑 ---
    # 左图：高对比度 Viridis
    left_img = cv2.applyColorMap(spatial_base, cv2.COLORMAP_VIRIDIS)
    
    curr_uid = int(unit_id) if unit_id is not None else None
    
    # 高亮选中细胞
    if curr_uid is not None and curr_uid in cache["unit_ids"]:
        mask = (unit_id_map == curr_uid)
        if np.any(mask):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(left_img, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)

    # 排除细胞变暗
    for ex_uid in excluded_units:
        m = (unit_id_map == int(ex_uid))
        if np.any(m):
            left_img[m] = (left_img[m] * 0.2).astype(np.uint8)

    # 右图：C 曲线
    right_img = np.zeros((target_h, curve_w, 3), dtype=np.uint8)
    if curr_uid is not None and curr_uid in traces:
        trace = traces[curr_uid]
        if trace.size > 0:
            tx = np.linspace(10, curve_w-10, len(trace)).astype(np.int32)
            t_min, t_max = np.min(trace), np.max(trace)
            if t_max > t_min:
                ty = (target_h-30) - ((trace - t_min) / (t_max - t_min + 1e-8) * (target_h-60)).astype(np.int32)
                pts = np.stack([tx, ty], axis=1).reshape((-1, 1, 2))
                cv2.polylines(right_img, [pts], False, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(right_img, f"Unit: {curr_uid}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

    # 合成
    left_w = int(target_h * aspect)
    left_resized = cv2.resize(left_img, (left_w, target_h), interpolation=cv2.INTER_LINEAR)
    unit_map_resized = cv2.resize(unit_id_map, (left_w, target_h), interpolation=cv2.INTER_NEAREST)

    combined = np.hstack([left_resized, right_img])

    return {
        "image": combined,
        "unit_id_map": unit_map_resized,
        "left_width": left_w,
        "used_unit_id": curr_uid,
        "cache": cache
    }
    
def create_spatial_exploration_plot(result_dict: dict, penalty: float) -> np.ndarray:
    """
    绘制 first_spatial_update_explore 的单参数可视化：
    - 左上: Spatial Matrix: binary
    - 左下: Spatial Matrix: pseudo-color
    - 右侧: Temporal Components
    - 右下角文本框: 日志输出
    """
    print(f"执行: 生成 sparse_penalty={penalty} 的空间探索图...")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    A_sample = result_dict["A_sample"]
    C_sample = result_dict["C_sample"]
    log_lines = result_dict.get("log_lines", [])

    # ---- 计算空间图 ----
    if isinstance(A_sample, xr.DataArray):
        A_bin = (A_sample > 0).sum("unit_id").compute().values.astype(np.float32)
        A_pseudo = A_sample.max("unit_id").compute().values.astype(np.float32)
    else:
        # 兜底：假设是 numpy，形状 (unit, h, w)
        A_bin = (A_sample > 0).sum(axis=0).astype(np.float32)
        A_pseudo = A_sample.max(axis=0).astype(np.float32)

    # ---- 计算 temporal ----
    if isinstance(C_sample, xr.DataArray):
        C_arr = C_sample.compute().values
        # 如果是 (frame, unit)，转成 (unit, frame)
        if C_sample.dims[0] == "frame":
            C_arr = C_arr.T
    else:
        C_arr = np.asarray(C_sample)
        # 简单兜底判断
        if C_arr.shape[0] > C_arr.shape[1]:
            pass

    fig = plt.figure(figsize=(12, 7), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    # 左上：binary
    im1 = ax1.imshow(A_bin, cmap="viridis")
    ax1.set_title("Spatial Matrix: binary", fontsize=13, fontweight="bold")
    ax1.set_xlabel("width", fontstyle="italic")
    ax1.set_ylabel("height", fontstyle="italic")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 左下：pseudo-color
    im2 = ax2.imshow(A_pseudo, cmap="viridis")
    ax2.set_title("Spatial Matrix: pseudo-color", fontsize=13, fontweight="bold")
    ax2.set_xlabel("width", fontstyle="italic")
    ax2.set_ylabel("height", fontstyle="italic")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 右侧：Temporal Components
    ax3.set_title("Temporal Components", fontsize=18, fontweight="bold")
    n_show = min(10, C_arr.shape[0])
    offset = 0.0

    for i in range(n_show):
        cur = np.asarray(C_arr[i]).astype(np.float32)
        if np.all(~np.isfinite(cur)):
            continue
        cur = np.nan_to_num(cur)
        ax3.plot(cur + offset, linewidth=0.6)
        local_span = max(cur.max() - cur.min(), 1.0)
        offset += local_span * 1.2

    ax3.set_xlabel("time", fontstyle="italic")
    ax3.set_yticks([])

    # 日志框
    log_text = "\n".join(log_lines)
    ax3.text(
        0.98, 0.02, log_text,
        transform=ax3.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray")
    )

    fig.suptitle(f"sparse_penalty = {penalty}", fontsize=14)
    fig.tight_layout()
    return fig_to_rgb_array(fig)

def create_spatial_exploration_compare_plot(
    left_result: dict,
    right_result: dict,
    left_penalty: float,
    right_penalty: float
) -> np.ndarray:
    """
    绘制 first_spatial_update_explore 的双参数对比图：
    左右分别显示两个 sparse_penalty 对应的 spatial binary / pseudo-color。
    """
    print(f"执行: 生成双参数空间对比图 ({left_penalty} vs {right_penalty}) ...")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    def _get_spatial_maps(result_dict):
        A_sample = result_dict["A_sample"]
        if isinstance(A_sample, xr.DataArray):
            A_bin = (A_sample > 0).sum("unit_id").compute().values.astype(np.float32)
            A_pseudo = A_sample.max("unit_id").compute().values.astype(np.float32)
        else:
            A_bin = (A_sample > 0).sum(axis=0).astype(np.float32)
            A_pseudo = A_sample.max(axis=0).astype(np.float32)
        return A_bin, A_pseudo

    left_bin, left_pseudo = _get_spatial_maps(left_result)
    right_bin, right_pseudo = _get_spatial_maps(right_result)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)

    # 左上
    im = axes[0, 0].imshow(left_bin, cmap="viridis")
    axes[0, 0].set_title(f"Spatial Matrix: binary\npenalty = {left_penalty}")
    axes[0, 0].set_xlabel("width", fontstyle="italic")
    axes[0, 0].set_ylabel("height", fontstyle="italic")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 左下
    im = axes[1, 0].imshow(left_pseudo, cmap="viridis")
    axes[1, 0].set_title(f"Spatial Matrix: pseudo-color\npenalty = {left_penalty}")
    axes[1, 0].set_xlabel("width", fontstyle="italic")
    axes[1, 0].set_ylabel("height", fontstyle="italic")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 右上
    im = axes[0, 1].imshow(right_bin, cmap="viridis")
    axes[0, 1].set_title(f"Spatial Matrix: binary\npenalty = {right_penalty}")
    axes[0, 1].set_xlabel("width", fontstyle="italic")
    axes[0, 1].set_ylabel("height", fontstyle="italic")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 右下
    im = axes[1, 1].imshow(right_pseudo, cmap="viridis")
    axes[1, 1].set_title(f"Spatial Matrix: pseudo-color\npenalty = {right_penalty}")
    axes[1, 1].set_xlabel("width", fontstyle="italic")
    axes[1, 1].set_ylabel("height", fontstyle="italic")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle("Spatial Comparison", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig_to_rgb_array(fig)


def create_temporal_exploration_plot(result_dict: dict, penalty: float) -> np.ndarray:
    """绘制 temporal explore 单参数结果：完整时间轴多曲线叠加（raw/C_after/fit/spike）。"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    c_after = np.asarray(result_dict.get("c_after_trace", []), dtype=np.float32)
    fit = np.asarray(result_dict.get("fit_trace", []), dtype=np.float32)
    spikes = np.asarray(result_dict.get("spike_trace", []), dtype=np.float32)
    focus_unit = result_dict.get("focus_unit", "N/A")
    log_lines = result_dict.get("log_lines", [])

    # 兼容旧结果结构（若新字段缺失）
    if c_after.size == 0 or fit.size == 0 or spikes.size == 0:
        c_after = result_dict.get("C_after")
        s_after = result_dict.get("S_after")
        if isinstance(c_after, xr.DataArray):
            c_after = c_after.compute().values
        if isinstance(s_after, xr.DataArray):
            s_after = s_after.compute().values
        c_after = np.asarray(c_after, dtype=np.float32)
        s_after = np.asarray(s_after, dtype=np.float32)
        if c_after.ndim == 2 and c_after.shape[0] > 0:
            fit = c_after[0]
            c_after = c_after[0]
        if s_after.ndim == 2 and s_after.shape[0] > 0:
            spikes = s_after[0]

    raw_mc = np.asarray(result_dict.get("raw_mc_trace", []), dtype=np.float32)

    def _target_len() -> int:
        # 优先使用 raw_mc 全长；若缺失则用可用信号的最长长度
        c_before = result_dict.get("C_before")
        n_c_before = 0
        if isinstance(c_before, xr.DataArray):
            n_c_before = int(c_before.sizes.get("frame", 0))
        elif isinstance(c_before, np.ndarray) and c_before.ndim >= 2:
            n_c_before = int(c_before.shape[-1])

        candidates = [
            int(getattr(raw_mc, "size", 0)),
            int(getattr(c_after, "size", 0)),
            int(getattr(fit, "size", 0)),
            int(getattr(spikes, "size", 0)),
            n_c_before,
        ]
        n = int(max(candidates))
        return max(1, n)

    def _full(x: np.ndarray, n: int, fallback=None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0 and fallback is not None:
            x = np.asarray(fallback, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return np.zeros((n,), dtype=np.float32)
        if x.size >= n:
            return x[:n]
        pad = np.zeros((n - x.size,), dtype=np.float32)
        return np.concatenate([x, pad], axis=0)

    n = _target_len()
    c_after = _full(c_after, n)
    fit = _full(fit, n, fallback=c_after)
    spikes = _full(spikes, n)
    raw_mc = _full(raw_mc, n, fallback=c_after)
    t = np.arange(n)

    def _norm(x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x.astype(np.float32))
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        if hi - lo < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return (x - lo) / (hi - lo)

    raw_n = _norm(raw_mc)
    c_n = _norm(c_after)
    fit_n = _norm(fit)
    spk_n = _norm(np.abs(spikes))

    fig = plt.figure(figsize=(14, 6), dpi=110)
    ax = fig.add_subplot(1, 1, 1)

    # 全部按完整时间轴绘制（与示例图一致）
    ax.plot(t, c_n, label='Fitted Calcium Trace', color='#1f77b4', linewidth=1.1, alpha=0.9)
    ax.plot(t, fit_n, label='Fitted Signal', color='#ff7f0e', linewidth=1.1, alpha=0.9)
    ax.plot(t, spk_n, label='Fitted Spikes', color='#2ca02c', linewidth=0.9, alpha=0.8)
    ax.plot(t, raw_n, label='Raw Signal', color='#d62728', linewidth=0.9, alpha=0.75)

    ax.set_title(f"Current Unit: Temporal Traces (unit={focus_unit}, penalty={penalty})", fontsize=13, fontweight='bold')
    ax.set_xlabel('frame', fontstyle='italic')
    ax.set_ylabel('Intensity (A.U.)', fontstyle='italic')
    ax.grid(alpha=0.22)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    log_text = "\n".join(log_lines)
    ax.text(
        0.02,
        0.02,
        log_text,
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=9,
        family='monospace',
        bbox=dict(facecolor='white', alpha=0.75, edgecolor='gray'),
    )

    fig.suptitle(f"Temporal Explore · sparse_penalty = {penalty}", fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig_to_rgb_array(fig)


def create_temporal_exploration_compare_plot(
    left_result: dict,
    right_result: dict,
    left_penalty: float,
    right_penalty: float,
) -> np.ndarray:
    """绘制 temporal explore 双参数对比：上下两图展示两个 penalty 的完整时间轴多曲线。"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    def _sig(res: dict):
        c_after = np.asarray(res.get("c_after_trace", []), dtype=np.float32)
        raw = np.asarray(res.get("raw_mc_trace", []), dtype=np.float32)
        fit = np.asarray(res.get("fit_trace", []), dtype=np.float32)
        spk = np.asarray(res.get("spike_trace", []), dtype=np.float32)

        c_before = res.get("C_before")
        n_c_before = 0
        if isinstance(c_before, xr.DataArray):
            n_c_before = int(c_before.sizes.get("frame", 0))
        elif isinstance(c_before, np.ndarray) and c_before.ndim >= 2:
            n_c_before = int(c_before.shape[-1])

        n = int(max(len(c_after), len(fit), len(spk), len(raw), n_c_before, 1))

        def _full(x: np.ndarray, n_: int, fallback=None) -> np.ndarray:
            x = np.asarray(x, dtype=np.float32).reshape(-1)
            if x.size == 0 and fallback is not None:
                x = np.asarray(fallback, dtype=np.float32).reshape(-1)
            if x.size == 0:
                return np.zeros((n_,), dtype=np.float32)
            if x.size >= n_:
                return x[:n_]
            return np.concatenate([x, np.zeros((n_ - x.size,), dtype=np.float32)], axis=0)

        c_after = _full(c_after, n)
        fit = _full(fit, n, fallback=c_after)
        spk = _full(spk, n)
        raw = _full(raw, n, fallback=c_after)
        return c_after, fit, spk, raw

    def _norm(x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x.astype(np.float32))
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        if hi - lo < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return (x - lo) / (hi - lo)

    lc, lfit, lspk, lraw = _sig(left_result)
    rc, rfit, rspk, rraw = _sig(right_result)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=110, sharex=False)

    lt = np.arange(len(lc))
    rt = np.arange(len(rc))

    axes[0].plot(lt, _norm(lc), color='#1f77b4', linewidth=1.0, alpha=0.9, label='Fitted Calcium Trace')
    axes[0].plot(lt, _norm(lfit), color='#ff7f0e', linewidth=1.0, alpha=0.9, label='Fitted Signal')
    axes[0].plot(lt, _norm(np.abs(lspk)), color='#2ca02c', linewidth=0.9, alpha=0.8, label='Fitted Spikes')
    axes[0].plot(lt, _norm(lraw), color='#d62728', linewidth=0.9, alpha=0.75, label='Raw Signal')
    axes[0].set_title(f"Top: penalty={left_penalty}")
    axes[0].set_xlabel('frame')
    axes[0].set_ylabel('Intensity (A.U.)')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].plot(rt, _norm(rc), color='#1f77b4', linewidth=1.0, alpha=0.9, label='Fitted Calcium Trace')
    axes[1].plot(rt, _norm(rfit), color='#ff7f0e', linewidth=1.0, alpha=0.9, label='Fitted Signal')
    axes[1].plot(rt, _norm(np.abs(rspk)), color='#2ca02c', linewidth=0.9, alpha=0.8, label='Fitted Spikes')
    axes[1].plot(rt, _norm(rraw), color='#d62728', linewidth=0.9, alpha=0.75, label='Raw Signal')
    axes[1].set_title(f"Bottom: penalty={right_penalty}")
    axes[1].set_xlabel('frame')
    axes[1].set_ylabel('Intensity (A.U.)')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(alpha=0.2)

    fig.suptitle(f"Temporal Explore Compare ({left_penalty} vs {right_penalty})", fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig_to_rgb_array(fig)

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

    # 使用宽屏比例，让图像在 UI 中更容易“铺满”可视区
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 9),
        dpi=120,
        facecolor="#171A21",
        constrained_layout=True,
    )
    
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
    cmaps = ['turbo', 'magma', 'turbo', 'magma']
    
    for i, ax in enumerate(axes.flat):
        data = plot_data[i]
        vmin, vmax = plot_ranges[i]
        
        # 确保数据不为空
        if data.size == 0:
            ax.set_title(f"{titles[i]} (无数据)")
            ax.axis('off')
            continue
            
        # 稳健对比度：避免极端值导致画面发灰
        try:
            if np.isfinite(data).any():
                lo = float(np.nanpercentile(data, 1))
                hi = float(np.nanpercentile(data, 99))
                if hi > lo:
                    vmin, vmax = lo, hi
        except Exception:
            pass

        im = ax.imshow(data, cmap=cmaps[i], vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(titles[i], color="#E6EAF2", fontsize=11, pad=8)
        ax.axis('off')
        ax.set_facecolor("#171A21")

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
        cbar.outline.set_edgecolor("#8892A6")
        cbar.ax.yaxis.set_tick_params(color="#C9D1D9", labelsize=8)
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_color("#C9D1D9")

    fig.suptitle(
        f"{step_name} Spatial Update · Full Field View",
        color="#F3F6FC",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    
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
    step_name: str,
    dropped_examples: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    创建 CNMF 时间更新的矩阵 2x2 对比图：
    1. 初始 C 矩阵 (C_init)
    2. dropped unit 曲线示例（右上）
    3. 更新后 C 矩阵 (C_new)
    4. 更新后 S 矩阵 (S_new)

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
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=100)
    
    # 统一到 (unit, frame) 后再画图，保证 x=frame, y=unit。
    # 兼容旧调用：若输入是 (frame, unit) 则自动转置。
    def _as_unit_frame(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            return arr.reshape(1, -1)
        # 常见场景 frame 数远大于 unit 数，因此 shape[0] > shape[1] 时视为 (frame, unit)
        if arr.shape[0] > arr.shape[1]:
            return arr.T
        return arr

    C_init_T = _as_unit_frame(C_init)
    C_new_T = _as_unit_frame(C_new)
    S_new_T = _as_unit_frame(S_new)
    
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

    # Plot 2: dropped unit 曲线示例（右上）
    ax = axes[0, 1]
    ax.set_title(f"Dropped Unit Examples ({step_name})")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized Trace")
    has_dropped = False
    if isinstance(dropped_examples, dict):
        before_ls = dropped_examples.get("before", []) or []
        after_ls = dropped_examples.get("after", []) or []
        uid_ls = dropped_examples.get("unit_ids", []) or []
        n_show = min(len(before_ls), max(len(after_ls), len(before_ls)), 3)
        for i in range(n_show):
            b = np.asarray(before_ls[i], dtype=np.float32).reshape(-1)
            if i < len(after_ls):
                a = np.asarray(after_ls[i], dtype=np.float32).reshape(-1)
            else:
                a = np.zeros_like(b, dtype=np.float32)
            n = int(max(b.size, a.size, 1))
            if b.size < n:
                b = np.concatenate([b, np.zeros((n - b.size,), dtype=np.float32)])
            if a.size < n:
                a = np.concatenate([a, np.zeros((n - a.size,), dtype=np.float32)])

            def _norm1(x: np.ndarray) -> np.ndarray:
                x = np.nan_to_num(x)
                lo = float(np.min(x))
                hi = float(np.max(x))
                if hi - lo < 1e-8:
                    return np.zeros_like(x)
                return (x - lo) / (hi - lo)

            t = np.arange(n)
            uid_txt = uid_ls[i] if i < len(uid_ls) else f"#{i}"
            ax.plot(t, _norm1(b), linewidth=1.0, alpha=0.9, label=f"u{uid_txt} before")
            ax.plot(t, _norm1(a), linewidth=1.0, alpha=0.8, linestyle="--", label=f"u{uid_txt} after")
            has_dropped = True

    if not has_dropped:
        ax.text(0.5, 0.5, "No dropped units", ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        if has_dropped:
            ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    # Plot 3: Temporal Trace New
    ax = axes[1, 0]
    im_c = ax.imshow(C_new_T, aspect='auto', cmap='viridis', vmin=vmin_c, vmax=vmax_c)
    ax.set_title(f"Temporal Trace Updated ({step_name})")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Unit ID")
    fig.colorbar(im_c, ax=ax, fraction=0.046, pad=0.04)

    # Plot 4: Spikes New
    ax = axes[1, 1]
    im_s = ax.imshow(S_new_T, aspect='auto', cmap='magma', vmin=vmin_s, vmax=vmax_s)
    ax.set_title(f"Spikes Updated ({step_name})")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Unit ID")
    fig.colorbar(im_s, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    return fig_to_rgb_array(fig)


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
    
    # 统一到 (unit, frame) 后再画图，保证 x=frame, y=unit。
    def _as_unit_frame(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            return arr.reshape(1, -1)
        if arr.shape[0] > arr.shape[1]:
            return arr.T
        return arr

    C_before_T = _as_unit_frame(C_before)
    C_after_T = _as_unit_frame(C_after)
    
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
    
    return fig_to_rgb_array(fig)

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