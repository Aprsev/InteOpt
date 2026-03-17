import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple


# =========================
# 配置区：哪些包优先走 conda/mamba
# =========================
CONDA_FIRST_PACKAGES = {
    # 科学计算 / 编译型 / Windows 常见易炸包
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "scikit-image",
    "numba",
    "netcdf4",
    "xarray",
    "zarr",
    "opencv-python",
    "opencv",
    "pyfftw",
    "simpleitk",
    "medpy",
    "pyqt5",
    "pyqt",
    "pymetis",
    "h5py",
    "hdf5",
    "matplotlib",
    "pillow",
    "lxml",
    "pyyaml",
    "tifffile",
    # jupyter 生态在 Windows 上也更建议 conda
    "jupyter",
    "jupyterlab",
    "ipykernel",
    "notebook",
    "jupyter-server",
    "pywinpty",
}

# pip 包名到 conda 包名的常见映射
CONDA_NAME_MAP = {
    "opencv-python": "opencv",
    "pyqt5": "pyqt",
    "jupyter-server": "jupyter_server",
}


# =========================
# 终端样式
# =========================
class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"


def supports_color() -> bool:
    return sys.stdout.isatty()


def color(text: str, style: str) -> str:
    if not supports_color():
        return text
    return f"{style}{text}{Style.RESET}"


def print_header(title: str):
    line = "=" * 72
    print(color(f"\n{line}\n{title}\n{line}", Style.BOLD + Style.CYAN))


def print_section(title: str):
    print(color(f"\n--- {title} ---", Style.BOLD + Style.BLUE))


def print_ok(msg: str):
    print(color(f"[OK] {msg}", Style.GREEN))


def print_warn(msg: str):
    print(color(f"[WARN] {msg}", Style.YELLOW))


def print_err(msg: str):
    print(color(f"[ERR] {msg}", Style.RED))


def print_info(msg: str):
    print(color(f"[INFO] {msg}", Style.CYAN))


# =========================
# 数据结构
# =========================
@dataclass
class InstallResult:
    name: str
    method: str
    success: bool
    message: str = ""
    duration_sec: float = 0.0


@dataclass
class Summary:
    conda_success: List[InstallResult] = field(default_factory=list)
    conda_failed: List[InstallResult] = field(default_factory=list)
    pip_success: List[InstallResult] = field(default_factory=list)
    pip_failed: List[InstallResult] = field(default_factory=list)


# =========================
# 工具函数
# =========================
REQ_LINE_RE = re.compile(
    r"^\s*([A-Za-z0-9_.\-]+)\s*([<>=!~].*)?$"
)


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def parse_requirements(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"requirements file not found: {path}")

    pkgs = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            if not line or line.startswith("#"):
                continue
            if line.startswith("-e ") or line.startswith("--"):
                continue
            if "://" in line or line.startswith("git+"):
                pkgs.append(line)
                continue

            m = REQ_LINE_RE.match(line)
            if m:
                pkgs.append(line)
            else:
                # 保留原样，后面交给 pip 兜底
                pkgs.append(line)

    return pkgs


def split_requirements(reqs: List[str]) -> Tuple[List[str], List[str]]:
    conda_list = []
    pip_list = []

    for req in reqs:
        if "://" in req or req.startswith("git+"):
            pip_list.append(req)
            continue

        m = REQ_LINE_RE.match(req)
        if not m:
            pip_list.append(req)
            continue

        raw_name = m.group(1)
        norm = normalize_name(raw_name)

        if norm in CONDA_FIRST_PACKAGES:
            # conda 名称可能不同
            conda_name = CONDA_NAME_MAP.get(norm, norm)
            version_part = req[len(raw_name):].strip()
            conda_req = f"{conda_name}{version_part}" if version_part else conda_name
            conda_list.append(conda_req)
        else:
            pip_list.append(req)

    return conda_list, pip_list


def run_cmd(cmd: List[str], short_name: str) -> Tuple[bool, str, float]:
    start = time.time()
    print_info(f"执行: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
        duration = time.time() - start
        output = proc.stdout.strip()

        if proc.returncode == 0:
            return True, output, duration
        return False, output, duration

    except Exception as e:
        duration = time.time() - start
        return False, str(e), duration


def ensure_env_activated():
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        raise RuntimeError(
            "未检测到已激活的 conda 环境。请先执行 `conda activate <env_name>` 再运行脚本。"
        )
    return prefix


def choose_conda_exe(preferred: str = None) -> str:
    if preferred:
        exe = shutil.which(preferred)
        if exe:
            return preferred
        raise RuntimeError(f"找不到指定的 conda 可执行程序: {preferred}")

    for exe in ("mamba", "conda"):
        if shutil.which(exe):
            return exe

    raise RuntimeError("未找到 mamba 或 conda，请确认它们已加入 PATH。")


def install_with_conda(conda_exe: str, channel: str, packages: List[str]) -> List[InstallResult]:
    results = []
    if not packages:
        return results

    print_section("Conda/Mamba 安装阶段")

    total = len(packages)
    for i, pkg in enumerate(packages, 1):
        print(color(f"[{i}/{total}] conda 安装: {pkg}", Style.BOLD))
        cmd = [conda_exe, "install", "-y", "-c", channel, pkg]
        ok, output, sec = run_cmd(cmd, pkg)

        if ok:
            print_ok(f"{pkg} 安装成功 ({sec:.1f}s)")
            results.append(InstallResult(pkg, "conda", True, duration_sec=sec))
        else:
            print_err(f"{pkg} 安装失败 ({sec:.1f}s)")
            tail = "\n".join(output.splitlines()[-15:]) if output else ""
            if tail:
                print(color(tail, Style.DIM))
            results.append(InstallResult(pkg, "conda", False, message=output, duration_sec=sec))

    return results


def install_with_pip(packages: List[str]) -> List[InstallResult]:
    results = []
    if not packages:
        return results

    print_section("pip 安装阶段")

    total = len(packages)
    for i, pkg in enumerate(packages, 1):
        print(color(f"[{i}/{total}] pip 安装: {pkg}", Style.BOLD))
        cmd = [sys.executable, "-m", "pip", "install", pkg]
        ok, output, sec = run_cmd(cmd, pkg)

        if ok:
            print_ok(f"{pkg} 安装成功 ({sec:.1f}s)")
            results.append(InstallResult(pkg, "pip", True, duration_sec=sec))
        else:
            print_err(f"{pkg} 安装失败 ({sec:.1f}s)")
            tail = "\n".join(output.splitlines()[-15:]) if output else ""
            if tail:
                print(color(tail, Style.DIM))
            results.append(InstallResult(pkg, "pip", False, message=output, duration_sec=sec))

    return results


def print_plan(conda_pkgs: List[str], pip_pkgs: List[str], conda_exe: str, channel: str):
    print_header("安装计划预览")
    print_info(f"操作系统: {platform.system()} {platform.release()}")
    print_info(f"Python: {sys.version.split()[0]}")
    print_info(f"当前环境: {os.environ.get('CONDA_DEFAULT_ENV', '(unknown)')}")
    print_info(f"环境路径: {os.environ.get('CONDA_PREFIX', '(unknown)')}")
    print_info(f"Conda执行器: {conda_exe}")
    print_info(f"Conda频道: {channel}")

    print_section(f"Conda/Mamba 优先安装 ({len(conda_pkgs)} 个)")
    for pkg in conda_pkgs:
        print(f"  - {pkg}")

    print_section(f"pip 安装 ({len(pip_pkgs)} 个)")
    for pkg in pip_pkgs:
        print(f"  - {pkg}")


def print_summary(summary: Summary):
    print_header("安装结果汇总")

    def dump(title: str, items: List[InstallResult], ok: bool):
        if ok:
            print_ok(f"{title}: {len(items)}")
        else:
            if items:
                print_err(f"{title}: {len(items)}")
            else:
                print_ok(f"{title}: 0")

        for item in items:
            label = "OK" if item.success else "FAIL"
            print(f"  [{label}] {item.name} ({item.duration_sec:.1f}s)")
            if (not item.success) and item.message:
                lines = item.message.splitlines()
                tail = "\n".join(lines[-8:])
                if tail:
                    print(color(f"      {tail}", Style.DIM))

    dump("Conda 成功", summary.conda_success, True)
    dump("Conda 失败", summary.conda_failed, False)
    dump("pip 成功", summary.pip_success, True)
    dump("pip 失败", summary.pip_failed, False)

    total_ok = len(summary.conda_success) + len(summary.pip_success)
    total_fail = len(summary.conda_failed) + len(summary.pip_failed)

    print_section("最终结论")
    print_info(f"总成功: {total_ok}")
    print_info(f"总失败: {total_fail}")

    if total_fail == 0:
        print_ok("全部安装成功。")
    else:
        print_warn("存在失败包。建议优先检查失败列表中的最后几行错误信息。")


def main():
    parser = argparse.ArgumentParser(
        description="在已激活 conda 环境中，按 conda/pip 分流安装 requirements。"
    )
    parser.add_argument("requirements", help="requirements 文件路径，例如 environment.txt")
    parser.add_argument(
        "--conda-exe",
        default=None,
        help="指定 conda 执行器，例如 mamba 或 conda；默认优先 mamba",
    )
    parser.add_argument(
        "--channel",
        default="conda-forge",
        help="conda/mamba 使用的 channel，默认 conda-forge",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示分类和安装计划，不实际安装",
    )
    args = parser.parse_args()

    try:
        ensure_env_activated()
        conda_exe = choose_conda_exe(args.conda_exe)
        reqs = parse_requirements(args.requirements)
        conda_pkgs, pip_pkgs = split_requirements(reqs)

        print_plan(conda_pkgs, pip_pkgs, conda_exe, args.channel)

        if args.dry_run:
            print_warn("dry-run 模式：未执行安装。")
            return

        summary = Summary()

        conda_results = install_with_conda(conda_exe, args.channel, conda_pkgs)
        for r in conda_results:
            if r.success:
                summary.conda_success.append(r)
            else:
                summary.conda_failed.append(r)

        pip_results = install_with_pip(pip_pkgs)
        for r in pip_results:
            if r.success:
                summary.pip_success.append(r)
            else:
                summary.pip_failed.append(r)

        print_summary(summary)

        if summary.conda_failed or summary.pip_failed:
            sys.exit(1)

    except Exception as e:
        print_err(str(e))
        sys.exit(2)


if __name__ == "__main__":
    main()