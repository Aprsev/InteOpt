# ui_v1 环境配置与使用说明

本项目提供 `install.py` 用于在已激活的 Conda 环境中，按 **conda 优先 / pip 兜底** 的策略安装依赖。依赖清单位于 `environment.txt`。

## 1. 环境准备

1. 安装 Miniconda/Anaconda（建议已配置到 PATH）。
2. （可选）安装 mamba（更快）：
   - `conda install -n base -c conda-forge mamba`
3. 进入项目目录。

## 2. 创建并激活 Conda 环境

建议使用 Python 3.8：

```bash
conda create -n ui python=3.8
conda activate ui
```

> 说明：`install.py` 会检测是否已激活 Conda 环境，未激活将直接报错。

## 3. 安装依赖（推荐流程）

### 3.1 预览安装计划（不实际安装）

```bash
python install.py environment.txt --dry-run
```

该步骤会输出：
- 当前系统、Python 版本、Conda 环境信息
- Conda/Mamba 与 pip 的分流列表

### 3.2 执行安装

```bash
python install.py environment.txt
```

### 3.3 指定 Conda 执行器或频道

- 指定执行器（mamba/conda）：

```bash
python install.py environment.txt --conda-exe mamba
```

- 指定频道（默认 `conda-forge`）：

```bash
python install.py environment.txt --channel conda-forge
```

## 4. 安装策略说明

`install.py` 会：
- 优先使用 Conda/Mamba 安装易出错或编译类包（如 `numpy`、`scipy`、`pyqt` 等）；
- 其余包使用 pip 安装；
- 输出安装进度与结果汇总，失败时给出末尾错误信息。

包列表来源：`environment.txt`。

## 5. 常见问题

### 5.1 未检测到已激活的 Conda 环境

请先执行：

```bash
conda activate ui
```

### 5.2 找不到 conda 或 mamba

请确认已将 Conda 安装目录加入 PATH，或使用：

```bash
python install.py environment.txt --conda-exe conda
```

### 5.3 某些包安装失败

建议：
1. 查看失败包的错误末尾信息；
2. 尝试切换频道或使用 mamba；
3. 手动单独安装失败包进行排查。

## 6. 运行项目

完成依赖安装后，按项目入口文件运行（示例）：

```bash
python main.py
```

如需其他入口，请查看项目内相关脚本。
