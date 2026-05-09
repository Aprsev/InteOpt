# 测试记录（UI V1）

## 1. 单步骤测试：first_spatial_update_explore

- 测试代码：[test/test_first_spatial_update_explore.py](test/test_first_spatial_update_explore.py)
- 目标功能：
	- 仅测试 first_spatial_update_explore 步骤
	- 检查前序结果缓存读取（有缓存则不重跑）
	- 检查返回格式与数据有效性
	- 检查内部格式转换（DataArray -> DataFrame）
	- 打开 MainWindow 进行可视化，并支持手动判定通过/不通过
	- 提供置顶小窗显示前序步骤进度与测试点进度
- 实现情况：
	- 已实现并可执行（最近一次单步测试命令返回成功）
	- 已支持失败即停、错误去重、终端输出数据快照

## 2. 多参数可视化切换/对比测试

- 测试代码：[test/test_first_spatial_update_explore_multi_view.py](test/test_first_spatial_update_explore_multi_view.py)
- 目标功能：
	- 验证在 first_spatial_update_explore 中多 sparse_penalty 的单参数切换
	- 验证双参数自主选择并应用对比显示
	- 验证 MainWindow 中对应控件显示与可视化结果可渲染
- 依赖的 UI 功能实现：
	- [main_pipeline_window.py](main_pipeline_window.py) 已增加：
		- 单参数切换按钮（上一个参数 / 下一个参数）
		- 双参数对比应用按钮
		- 参数选择状态保留与容错匹配逻辑
- 实现情况：
	- 测试文件已创建并修复导入冲突
	- 当前状态：已通过（最近一次 `python .\test_first_spatial_update_explore_multi_view.py` 返回成功）

## 3. 第二次空间探索（与第一次一致）改造

- 代码改造：
	- [minian_processor.py](minian_processor.py)
		- 新增 `run_second_spatial_update_explore()`
		- 输入改为第二轮起点：`A_iter1_merged` / `C_iter1_merged`
		- 复用与第一次相同的探索逻辑：多 `sparse_penalty`、单/双参数可视化结果结构、鲁棒性修复
	- [main_pipeline_window.py](main_pipeline_window.py)
		- 流程新增步骤：`second_spatial_update_explore`（exploration）
		- 探索控制区改为同时支持第一/第二次空间探索
	- [default_config.json](default_config.json)
		- 新增 `second_spatial_update_explore` 参数块，默认与 `first_spatial_update_explore` 一致
- 功能目标：
	- 第二次空间探索在交互体验与结果结构上与第一次保持一致
	- 同样支持：参数切换、双参数对比、可视化模式切换

## 4. 建议执行命令

- 单步骤测试：
	- `python .\test\test_first_spatial_update_explore.py`
- 多参数切换/对比测试：
	- `python .\test\test_first_spatial_update_explore_multi_view.py`

## 5. Spatial Update 执行调试与整视野可视化一致性

- 代码改造：
	- [minian_processor.py](minian_processor.py)
		- 新增 `run_first_spatial_update_exec()`
		- `run_second_spatial_update()` 改为与 explore/exec 一致的参数与数据预处理流程
		- 两个执行步骤都使用与 explore 一致的输入对齐、sn_spatial 修复、frame 单块、NaN/Inf 清理
		- 执行步骤可视化改为整视野 2x2 空间图（非单 unit）并缓存到 `*_vis_array`
	- [main_pipeline_window.py](main_pipeline_window.py)
		- `cnmf_update` 分支中：`first_spatial_update_exec` 与 `second_spatial_update` 优先显示 `*_vis_array`（整视野）
		- 其他 `cnmf_update` 步骤保持原逻辑

- 新增测试代码：
	- [test/test_first_spatial_update_exec_fullfield.py](test/test_first_spatial_update_exec_fullfield.py)
	- 测试功能：
		- 先运行 `first_spatial_update_explore`，再运行 `first_spatial_update_exec`
		- 验证生成整视野可视化数组
		- 打开 MainWindow 渲染并支持人工确认整视野效果

- 建议执行：
	- `python .\test\test_first_spatial_update_exec_fullfield.py`

## 6. First/Second Spatial Update 执行逻辑完全一致化

- 代码改造：
	- [minian_processor.py](minian_processor.py)
		- 新增统一执行入口：`_run_spatial_update_exec_common(...)`
		- `run_first_spatial_update_exec()` 与 `run_second_spatial_update()` 仅保留以下差异：
			- 探索步骤参数来源（`first_spatial_update_explore` / `second_spatial_update_explore`）
			- 输入键（`A_init,C_init` / `A_iter1_merged,C_iter1_merged`）
			- 输出键（`*_iter1` / `*_iter2`）
			- 回退背景时间项键（`f_init` / `f_iter1`）
			- 标题文案（First/Second Update）
		- 其余流程完全一致：
			- 参数融合
			- `sparse_penalty` 解析
			- `update_spatial` 调用
			- `C/C_chk/b/f` 生成与保存
			- 整视野 2x2 可视化生成与缓存

- 新增测试代码：
	- [test/test_second_spatial_update_exec_fullfield.py](test/test_second_spatial_update_exec_fullfield.py)
	- 测试功能：
		- 准备 second 执行输入链路
		- 执行 `second_spatial_update_explore` + `second_spatial_update`
		- 验证整视野可视化缓存与 MainWindow 渲染

- 当前状态：
	- `first_spatial_update_exec` 测试已完成（最近一次命令返回成功）
	- `second_spatial_update` 已完成代码同步与测试脚本补齐
	- 两步执行流程现已由同一公共函数驱动，除参数与输入输出键外无分叉逻辑

- 建议执行：
	- `python .\test\test_second_spatial_update_exec_fullfield.py`

## 7. Temporal Update Explore（多 sparse penalty + 双参数对比）

- 代码改造：
	- [minian_processor.py](minian_processor.py)
		- `run_first_temporal_update_explore()` 改为参数探索模式：
			- 对 `sparse_penalty_list`（兼容 `exploration_penalties`）逐一调用 `update_temporal`
			- 使用 CNMF 中 `compute_trace` + `update_temporal` 进行计算
			- 产出与 spatial explore 一致的结果结构：`mode/penalty_list/results/default_penalty`
	- [main_pipeline_window.py](main_pipeline_window.py)
		- 探索交互扩展到 temporal explore：
			- 参数切换（上一个/下一个）
			- 双参数对比（左/右 penalty）
		- exploration 渲染分支支持 temporal 单参数图与对比图
	- [minian_core/visualization.py](minian_core/visualization.py)
		- 新增 `create_temporal_exploration_plot()`
		- 新增 `create_temporal_exploration_compare_plot()`
		- 重点可视化目标：
			- 单细胞 `C_after 完整曲线` vs `拟合曲线`
			- 在拟合曲线上同步标记 `spike 时间`
			- 便于在当前 sparse penalty 下判断 spike 时间与拟合效果
	- [default_config.json](default_config.json)
		- 补充 `first_temporal_update_explore` 参数：
			- `sparse_penalty_list`
			- `p/add_lag/noise_freq/use_smooth/sample_units/random_seed`

- 新增测试代码：
	- [test/test_first_temporal_update_explore_multi_view.py](test/test_first_temporal_update_explore_multi_view.py)
	- 测试功能：
		- 校验 temporal explore 返回结构与关键字段有效性
		- 校验 `c_after_trace/fit_trace/spike_trace` 三类单细胞曲线数据
		- 校验单参数切换
		- 校验双参数对比渲染
		- 保留手动确认步骤（终端输入 `y`）

- 建议执行：
	- `python .\test\test_first_temporal_update_explore_multi_view.py`

- 隔离性修复（避免“第二次测试基于第一次结果”）：
	- [minian_processor.py](minian_processor.py)
		- temporal explore 输入改为深拷贝，并在每个 penalty 迭代使用独立副本，避免就地修改前序数据
	- [test/test_first_temporal_update_explore_multi_view.py](test/test_first_temporal_update_explore_multi_view.py)
		- 每次运行使用独立 `MINIAN_INTERMEDIATE` 子目录（带时间戳），避免复用上次测试产物

## 8. Spatial Update 关键结果矩阵保存（实际运行需求）

- 代码改造：
	- [minian_processor.py](minian_processor.py)
		- 在 `_run_spatial_update_exec_common(...)` 中补充并固定关键矩阵保存：
			- `A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1})`
			- `b = save_minian(b_new.rename("b"), intpath, overwrite=True)`
			- `f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)`
			- `C = save_minian(C_new.rename("C"), intpath, overwrite=True)`
			- `C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)`
		- 保存格式按默认 `save_minian` 流程，不增加额外格式变更。

- 新增测试代码：
	- [test/test_spatial_update_save_results.py](test/test_spatial_update_save_results.py)
	- 测试内容：
		1. 运行 `first_spatial_update_explore -> first_spatial_update_exec`
		2. 校验 repo 中存在 `A_iter1/b_iter1/f_iter1/C_iter1/C_chk_iter1`
		3. 校验中间目录存在 `A/b/f/C/C_chk` 保存产物
		4. 使用 `open_minian(..., return_dict=True)` 验证关键矩阵可读

- 建议执行：
	- `python .\test\test_spatial_update_save_results.py`

## 9. Temporal Update（单 penalty 执行 + merge + 同步骤可视化切换）

- 代码改造：
	- [minian_processor.py](minian_processor.py)
		- 新增统一执行逻辑 `_run_temporal_update_exec_common(...)`：
			- 计算并保存 `YrA`
			- 按“与 explore 一致”的参数来源执行 `update_temporal`（执行步仅取一个 penalty）
			- 保存 `C/C_chk/S/b0/c0` 关键矩阵
			- 执行 `unit_merge` 并保存 `A_iter*_merged/C_iter*_merged/C_chk_iter*_merged/sig_iter*_merged`
		- 新增 `run_first_temporal_update_exec()`，并保留 `run_first_temporal_update()` 兼容旧入口
		- `run_second_temporal_update()` 改为复用同一公共逻辑
	- [minian_core/visualization.py](minian_core/visualization.py)
		- `create_temporal_matrix_plot(...)` 升级为 2x2：
			- 左上：Temporal Trace Initial
			- 右上：dropped unit 曲线示例
			- 左下：Temporal Trace Updated
			- 右下：Spikes Updated
	- [main_pipeline_window.py](main_pipeline_window.py)
		- 新增 Temporal 结果切换控件（`update` / `merge`）
		- 在同一个 temporal update 步骤下切换显示 update 热图与 merge 热图

- 配置补充：
	- [default_config.json](default_config.json)
		- `first_temporal_update_exec`、`second_temporal_update` 增加：
			- `sparse_penalty`（单 penalty 执行入口）
			- `merge_kwargs`（merge 参数融合到 temporal update）

- 新增测试代码：
	- [test/test_temporal_update_exec_heatmap_merge_switch.py](test/test_temporal_update_exec_heatmap_merge_switch.py)
	- 测试内容：
		1. 执行 `first_temporal_update_exec`
		2. 校验关键矩阵保存：`YrA/C/C_chk/S/b0/c0`
		3. 校验 merge 结果写回：`A_iter1_merged/C_iter1_merged/C_chk_iter1_merged/sig_iter1_merged`
		4. 校验同步骤下 `update/merge` 视图可切换并成功渲染
		5. 最后一步增加 UI 手动确认（终端输入 `y`）
			- `update` 视图：确认 temporal update 热图（含 dropped unit 示例）
			- `merge` 视图：确认 merge 热图对比

- 建议执行：
	- `python .\test\test_temporal_update_exec_heatmap_merge_switch.py`

- 当前状态：
	- 已通过（最近一次 `python .\test\test_temporal_update_exec_heatmap_merge_switch.py` 返回成功）
	- 已验证：`update/merge` 视图切换可用，切换后不会自动回退到 `update`

## 10. 数据保存步骤（矩阵多选 + 保存格式 + 目录 + unit 过滤）

- 代码改造：
	- [minian_processor.py](minian_processor.py)
		- `run_save_data()` 升级为可配置保存：
			- `selected_matrices`：支持矩阵多选（如 `A/C/S/b/f/C_chk/sig`）
			- `save_format`：支持 `zarr/netcdf/csv/npy`
			- `output_dir`：支持自定义保存目录
			- `excluded_unit_ids`：支持手动剔除 unit 后再保存
		- 新增 `_resolve_save_data_inputs()`：按优先级自动解析可保存矩阵来源
		- 新增 `_apply_excluded_units()`：对含 `unit_id` 的矩阵执行过滤
	- [main_pipeline_window.py](main_pipeline_window.py)
		- 新增 save_data 交互区：
			- 矩阵多选列表（勾选）
			- 保存格式下拉框
			- 保存目录输入 + 浏览按钮
			- `unit_id` 选择框 + 排除当前 unit + 清空排除
		- 可视化区新增 save_data 交互预览（单个 unit 的空间分布 + 时间曲线）
	- [minian_core/visualization.py](minian_core/visualization.py)
		- 新增 `create_save_data_unit_plot(...)` 用于 save_data 交互预览
	- [default_config.json](default_config.json)
		- `save_data` 默认参数新增：
			- `selected_matrices`
			- `save_format`
			- `output_dir`
			- `excluded_unit_ids`

- 新增测试：
	- 前置链路固定为完整二次迭代：
		- `first_spatial_update_explore -> first_spatial_update_exec`
		- `first_temporal_update_explore -> first_temporal_update_exec`
		- `second_spatial_update_explore -> second_spatial_update`
		- `second_temporal_update_explore -> second_temporal_update`
	- 新增公共夹具：
		- [test/test_save_data_pipeline_fixture.py](test/test_save_data_pipeline_fixture.py)
		- 统一为 save_data 测试准备 second + temporal explore/update 后的数据上下文
	- [test/test_save_data_step_selection_and_filter.py](test/test_save_data_step_selection_and_filter.py)
		- 验证 CSV 保存 + unit 排除后不落盘
		- 验证 NPY 格式仅保存勾选矩阵
		- 验证全矩阵 `A/C/S/c0/b0/b/f` 可选并可保存
	- [test/test_save_data_ui_interaction.py](test/test_save_data_ui_interaction.py)
		- 验证 save_data 交互控件显示
		- 验证排除 unit 会更新配置
		- 验证预览图可渲染
		- 新增手动确认：矩阵多选/格式切换/unit 选择与排除交互

- 建议执行：
	- `python .\test\test_save_data_step_selection_and_filter.py`
	- `python .\test\test_save_data_ui_interaction.py`

