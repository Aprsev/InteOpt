import cv2
import numpy as np
import xarray as xr
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import uniform_filter
from skimage.morphology import disk
from scipy.signal import butter, filtfilt

from skimage.morphology import white_tophat

# --------------------------------------------------------------------------
# 新增辅助函数：用于时域滤波（FFT 滤波）
# --------------------------------------------------------------------------
def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    计算巴特沃斯带通滤波器的参数。
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter_per_pixel(varr: xr.DataArray, low_cut: float, high_cut: float, order: int, fs: float = 30.0) -> xr.DataArray:
    """
    对每个像素的时间序列应用带通滤波。
    """
    b, a = butter_bandpass(low_cut, high_cut, fs, order)

    # 重塑为 (time, pixels) 结构，方便 filtfilt 处理
    H, W = varr.sizes['height'], varr.sizes['width']
    T = varr.sizes['frame']
    
    # 转换为 NumPy 数组并重塑 (frame, H*W)
    arr = varr.compute().values.reshape(T, H * W)
    
    # 对每个像素应用滤波
    filtered_arr = np.zeros_like(arr)
    for i in range(H * W):
        # 确保数据类型为 float，因为 filtfilt 会使用 float
        filtered_arr[:, i] = filtfilt(b, a, arr[:, i].astype(np.float64))
        
    # 重塑回 (frame, height, width) 并转回 xarray
    filtered_arr = filtered_arr.reshape(T, H, W)
    
    # 创建新的 xr.DataArray
    res = xr.DataArray(
        filtered_arr,
        coords=varr.coords,
        dims=varr.dims,
        name=varr.name
    )
    return res


# --------------------------------------------------------------------------
# Minian 核心函数
# --------------------------------------------------------------------------

def remove_glow(varr: xr.DataArray, dpath: str, pattern: str) -> xr.DataArray:
    """
    通过从每一帧中减去每个像素的最小强度来去除视频中的光晕。
    ... (函数体不变) ...
    """
    print("正在执行：去除光晕...")
    # 计算每个像素在所有帧上的最小值
    min_vals = varr.min(dim="frame")
    # 从每一帧中减去该像素的最小值
    res = varr - min_vals
    print("去除光晕完成。")
    return res

def remove_glow(varr: xr.DataArray) -> xr.DataArray:
    """
    通过从每一帧中减去其所在像素在所有帧上的最小强度来去除视频中的光晕。

    参数
    ----------
    varr : xr.DataArray
        输入的视频数据，应包含维度 "height", "width" 和 "frame"。

    返回
    ------
    res : xr.DataArray
        去除光晕后的视频。与输入的 `varr` 具有相同的形状。
    """
    print("正在执行：去除光晕...")
    
    # 核心逻辑：计算每个像素在所有帧上的最小值
    min_vals = varr.min(dim="frame")
    
    # 从每一帧中减去该像素的最小值 (xarray 自动进行广播)
    res = varr - min_vals
    
    # 确保结果非负 (Minian 通常需要)
    res = res.clip(min=0) 
    
    print("去除光晕完成。")
    return res

def remove_background(varr: xr.DataArray, method: str, wnd: int) -> xr.DataArray:
    """
    Remove background from a video.
    ... (函数体不变) ...
    """
    selem = disk(wnd)
    res = xr.apply_ufunc(
        remove_background_perframe,
        varr,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[varr.dtype],
        kwargs=dict(method=method, wnd=wnd, selem=selem),
    )
    res = res.astype(varr.dtype)
    return res.rename(varr.name + "_subtracted")


def remove_background_perframe(
    fm: np.ndarray, method: str, wnd: int, selem: np.ndarray
) -> np.ndarray:
    """
    Remove background from a single frame.
    ... (函数体不变) ...
    """
    if method == "uniform":
        return fm - uniform_filter(fm, wnd)
    elif method == "tophat":
        # print("正在滤波")
        # print(cv2.morphologyEx(fm, cv2.MORPH_TOPHAT, selem))
        # return cv2.morphologyEx(fm, cv2.MORPH_TOPHAT, selem)
        fm32 = cv2.normalize(fm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        kernel = np.uint8(disk(wnd))  # 将布尔结构元转为0/1
        res = cv2.morphologyEx(fm32, cv2.MORPH_TOPHAT, kernel)
        res = res.astype(np.float32) / 255.0 * fm.max()  # 重新归一化
        # print("滤波完成")
        # print(res)
        return res
        # print("正在执行：白帽子滤波...")
        # print(white_tophat(fm, footprint=selem).shape)
        # return white_tophat(fm, footprint=selem)

    raise NotImplementedError(f"背景去除方法 {method} 不支持")


def stripe_correction(varr, reduce_dim="height", on="mean"):
    """
    Strip correction. (函数体不变)
    """
    if on == "mean":
        temp = varr.mean(dim="frame")
    elif on == "max":
        temp = varr.max(dim="frame")
    elif on == "perframe":
        temp = varr
    else:
        raise NotImplementedError("on {} not understood".format(on))
    mean1d = temp.mean(dim=reduce_dim)
    varr_sc = varr - mean1d
    return varr_sc.rename(varr.name + "_Stripe_Corrected")


def denoise(varr: xr.DataArray, method: str, **kwargs) -> xr.DataArray:
    """
    Denoise the movie frame by frame (Spatial) or pixel by pixel (Temporal).

    Parameters
    ----------
    varr : xr.DataArray
        The input movie data.
    method : str
        The method to use. Can be a spatial filter like 'gaussian' or 
        a temporal filter like 'fft' (for bandpass filtering).
    **kwargs
        Additional keyword arguments passed to the underlying functions.

    Returns
    -------
    res : xr.DataArray
        The resulting denoised movie.

    Raises
    ------
    NotImplementedError
        if the supplied `method` is not recognized
    """
    if method == "fft":
        print("正在执行：时域带通滤波 (FFT/Butterworth)...")
        # 时域滤波: 使用自定义的 apply_filter_per_pixel 函数
        return apply_filter_per_pixel(
            varr,
            low_cut=kwargs.get('low_cut', 0.1),
            high_cut=kwargs.get('high_cut', 0.5),
            order=kwargs.get('order', 5),
            fs=kwargs.get('fs', 30.0)
        ).rename(varr.name + "_filt")

    # 空间滤波
    elif method == "gaussian":
        func = cv2.GaussianBlur
    elif method == "anisotropic":
        func = anisotropic_diffusion
    elif method == "median":
        func = cv2.medianBlur
    elif method == "bilateral":
        func = cv2.bilateralFilter
    else:
        raise NotImplementedError("denoise method {} not understood".format(method))
        
    print(f"正在执行：空域滤波 ({method})...")

    res = xr.apply_ufunc(
        func,
        varr,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[varr.dtype],
        kwargs=kwargs,
    )
    res = res.astype(varr.dtype)
    return res.rename(varr.name + "_denoised")