import argparse
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# =========================
# 镜像源配置
# =========================
DEFAULT_CONDA_CHANNEL = "conda-forge"

CONDA_MIRRORS = {
    "tsinghua": {
        "conda-forge": "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge",
        "pkgs-main": "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main",
        "pkgs-msys2": "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2",
    },
    "ustc": {
        "conda-forge": "https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge",
        "pkgs-main": "https://mirrors.ustc.edu.cn/anaconda/pkgs/main",
        "pkgs-msys2": "https://mirrors.ustc.edu.cn/anaconda/pkgs/msys2",
    },
}

PIP_MIRRORS = {
    "tsinghua": "https://pypi.tuna.tsinghua.edu.cn/simple",
    "ustc": "https://mirrors.ustc.edu.cn/pypi/web/simple",
}


# =========================
# 哪些包优先走 conda/mamba
# =========================
CONDA_FIRST_PACKAGES = {
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
    "jupyter",
    "jupyterlab",
    "ipykernel",
    "notebook",
    "jupyter-server",
    "pywinpty",
}

CONDA_NAME_MAP = {
    "opencv-python": "opencv",
    "pyqt5": "pyqt",
    "jupyter-server": "jupyter_server",
}

# 必须使用 pip 安装的包（与实际环境保持一致）
PIP_ONLY_PACKAGES = {
    "opencv-python",
    "rtds-action",
    "setuptools-scm",
    "simpleitk",
    "medpy",
    "pyfftw",
    "netcdf4",
}

# Windows 下优先用 mamba 安装的必要包（尽量精简）
WINDOWS_MAMBA_PACKAGES = {
    "pyqt",
    "pyqt5",
    "pywinpty",
    "simpleitk",
    "medpy",
    "pyfftw",
    "pymetis",
    "opencv",
    "opencv-python",
}

# =========================
# conda 分组安装
# =========================
CONDA_INSTALL_GROUPS = [
    {
        "name": "基础数值计算",
        "packages": [
            "numpy",
            "scipy",
            "pandas",
            "matplotlib",
            "pillow",
            "pyyaml",
        ],
    },
    {
        "name": "机器学习与图像",
        "packages": [
            "scikit-learn",
            "scikit-image",
            "numba",
            "opencv",
            "opencv-python",
            "simpleitk",
            "medpy",
            "pyfftw",
            "tifffile",
        ],
    },
    {
        "name": "数据格式",
        "packages": [
            "netcdf4",
            "xarray",
            "zarr",
        ],
    },
    {
        "name": "Jupyter 生态",
        "packages": [
            "jupyter",
            "ipykernel",
            "jupyterlab",
            "pyqt",
            "pyqt5",
            "jupyter-server",
            "notebook",
        ],
    },
    {
        "name": "其他",
        "packages": [
            "pymetis",
        ],
    },
]


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
    MAGENTA = "\033[35m"


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


REQ_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*([<>=!~].*)?$")


# =========================
# 进度显示
# =========================
def make_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        total = 1
    ratio = min(max(current / total, 0), 1)
    filled = int(width * ratio)
    return "[" + "█" * filled + "-" * (width - filled) + "]"


def format_percent(current: int, total: int) -> str:
    if total <= 0:
        total = 1
    return f"{(current / total) * 100:6.2f}%"


class ProgressTracker:
    def __init__(self, total_packages: int):
        self.total_packages = max(total_packages, 1)
        self.completed = 0

    def advance(self, package_name: str, method: str, success: bool, status_label: Optional[str] = None):
        self.completed += 1
        bar = make_bar(self.completed, self.total_packages)
        pct = format_percent(self.completed, self.total_packages)
        if status_label is None:
            status_label = "SUCCESS" if success else "FAILED"
        status = color(status_label, Style.GREEN if success else Style.RED)
        print(color(
            f"{bar} {pct}:正在安装：{package_name} // {method} {status}",
            Style.BOLD
        ))

    def show_stage(self, title: str):
        bar = make_bar(self.completed, self.total_packages)
        pct = format_percent(self.completed, self.total_packages)
        print(color(
            f"{bar} {pct} | {title}",
            Style.BOLD + Style.MAGENTA
        ))


# =========================
# 基础工具
# =========================
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

        if norm in PIP_ONLY_PACKAGES:
            pip_list.append(req)
        else:
            conda_list.append(req)

    return sort_versioned_first(conda_list), sort_versioned_first(pip_list)


def extract_package_name(req: str) -> str:
    m = REQ_LINE_RE.match(req)
    if not m:
        return normalize_name(req)
    return normalize_name(m.group(1))


def has_version_spec(req: str) -> bool:
    if "://" in req or req.startswith("git+"):
        return True
    m = REQ_LINE_RE.match(req)
    if not m:
        return True
    return bool(m.group(2))


def sort_versioned_first(reqs: List[str]) -> List[str]:
    with_version = [r for r in reqs if has_version_spec(r)]
    without_version = [r for r in reqs if not has_version_spec(r)]
    return with_version + without_version


def group_conda_packages(packages: List[str]) -> List[Tuple[str, List[str]]]:
    grouped = []
    used = set()

    pkg_map = {}
    for pkg in packages:
        name = extract_package_name(pkg)
        mapped_name = normalize_name(CONDA_NAME_MAP.get(name, name))
        pkg_map[pkg] = mapped_name

    for group in CONDA_INSTALL_GROUPS:
        group_pkgs = []
        group_names = {normalize_name(x) for x in group["packages"]}

        for pkg in packages:
            if pkg in used:
                continue
            mapped_name = pkg_map[pkg]
            if mapped_name in group_names:
                group_pkgs.append(pkg)
                used.add(pkg)

        if group_pkgs:
            grouped.append((group["name"], group_pkgs))

    remaining = [pkg for pkg in packages if pkg not in used]
    if remaining:
        grouped.append(("未分组", remaining))

    return grouped


def ensure_env_activated():
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        raise RuntimeError(
            "未检测到已激活的 conda 环境。请先执行 `conda activate <env_name>` 再运行脚本。"
        )
    return prefix


def choose_conda_exe(preferred: Optional[str] = None) -> str:
    import shutil

    def prefer_exe(found_path: str) -> str:
        if not found_path:
            return found_path
        lower = found_path.lower()
        if not (lower.endswith(".bat") or lower.endswith(".cmd")):
            return found_path

        base = os.environ.get("CONDA_ROOT")
        if not base:
            conda_exe = os.environ.get("CONDA_EXE")
            if conda_exe:
                conda_exe = os.path.abspath(conda_exe)
                conda_dir = os.path.dirname(conda_exe)
                base = os.path.dirname(conda_dir)

        if not base:
            base = os.environ.get("CONDA_PREFIX")
            if base:
                base = os.path.abspath(base)

        if not base:
            return found_path

        candidates = [
            os.path.join(base, "Library", "bin", "mamba.exe"),
            os.path.join(base, "Scripts", "mamba.exe"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return found_path

    if preferred:
        if os.path.isfile(preferred):
            return preferred
        found = shutil.which(preferred)
        if found:
            return prefer_exe(found)
        raise RuntimeError(f"找不到指定的 conda 可执行程序: {preferred}")

    for name in ("mamba", "conda"):
        found = shutil.which(name)
        if found:
            return prefer_exe(found)

    base = os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_ROOT")
    if base:
        candidates = [
            os.path.join(base, "Library", "bin", "mamba.exe"),
            os.path.join(base, "Scripts", "mamba.exe"),
            os.path.join(base, "condabin", "mamba.bat"),
            os.path.join(base, "Scripts", "conda.exe"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

    raise RuntimeError("未找到 mamba 或 conda")


def tail_text(output: str, n: int = 20) -> str:
    if not output:
        return ""
    return "\n".join(output.splitlines()[-n:])


def filter_realtime_line(line: str) -> str:
    s = line.rstrip("\n")
    if not s.strip():
        return ""

    lower = s.lower()

    if "failed with initial frozen solve" in lower:
        return color(s, Style.YELLOW)
    if "solving environment" in lower:
        return color(s, Style.CYAN)
    if "collecting package metadata" in lower or "getting repodata from channels" in lower:
        return color(s, Style.CYAN)
    if "downloading and extracting packages" in lower:
        return color(s, Style.BLUE)
    if "preparing transaction" in lower:
        return color(s, Style.BLUE)
    if "verifying transaction" in lower:
        return color(s, Style.BLUE)
    if "executing transaction" in lower:
        return color(s, Style.BLUE)
    if "installing collected packages" in lower:
        return color(s, Style.BLUE)
    if "successfully installed" in lower:
        return color(s, Style.GREEN)
    if "error" in lower or "exception" in lower:
        return color(s, Style.RED)
    if "warning" in lower:
        return color(s, Style.YELLOW)

    return s


def run_cmd_live(
    cmd: List[str],
    title: str = "",
    show_live: bool = True,
    show_cmd: bool = True,
) -> Tuple[bool, str, float]:
    start = time.time()
    if show_cmd:
        print_info(f"执行: {' '.join(cmd)}")
    if show_live and title:
        print(color(f"[LIVE] {title}", Style.BOLD + Style.MAGENTA))

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            bufsize=1,
        )

        output_lines = []
        assert proc.stdout is not None

        for line in proc.stdout:
            output_lines.append(line)
            if show_live:
                pretty = filter_realtime_line(line)
                if pretty:
                    print(pretty, flush=True)

        proc.wait()
        duration = time.time() - start
        output = "".join(output_lines).strip()
        return proc.returncode == 0, output, duration

    except Exception as e:
        duration = time.time() - start
        return False, str(e), duration


def run_cmd_capture(cmd: List[str]) -> Tuple[bool, str]:
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
        return proc.returncode == 0, (proc.stdout or "")
    except Exception as e:
        return False, str(e)


def get_conda_method_name(conda_exe: str) -> str:
    exe_name = os.path.basename(conda_exe).lower()
    if "mamba" in exe_name:
        return "mamba"
    return "conda"


def build_conda_install_cmd(conda_exe: str, channel_args: List[str], pkg: str) -> List[str]:
    exe_lower = conda_exe.lower()
    base_cmd = [conda_exe, "install", "-y", *channel_args, pkg]

    if exe_lower.endswith(".bat") or exe_lower.endswith(".cmd"):
        return ["cmd", "/c", *base_cmd]
    return base_cmd


def build_conda_list_cmd(conda_exe: str) -> List[str]:
    exe_lower = conda_exe.lower()
    base_cmd = [conda_exe, "list", "--json"]
    if exe_lower.endswith(".bat") or exe_lower.endswith(".cmd"):
        return ["cmd", "/c", *base_cmd]
    return base_cmd


def get_conda_installed_map(conda_exe: str) -> dict:
    ok, output = run_cmd_capture(build_conda_list_cmd(conda_exe))
    if not ok:
        print_warn("无法读取 conda 已安装列表，将继续尝试安装所有包。")
        return {}

    try:
        import json

        data = json.loads(output)
        return {normalize_name(item.get("name", "")): item.get("version", "") for item in data}
    except Exception:
        print_warn("解析 conda 已安装列表失败，将继续尝试安装所有包。")
        return {}


def get_pip_installed_map() -> dict:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "list",
        "--format=json",
        "--disable-pip-version-check",
    ]
    ok, output = run_cmd_capture(cmd)
    if not ok:
        print_warn("无法读取 pip 已安装列表，将继续尝试安装所有包。")
        return {}

    try:
        import json

        data = json.loads(output)
        return {normalize_name(item.get("name", "")): item.get("version", "") for item in data}
    except Exception:
        print_warn("解析 pip 已安装列表失败，将继续尝试安装所有包。")
        return {}


# =========================
# 镜像与 channel
# =========================
def get_conda_channel_url(mirror_name: str, channel_name: str) -> str:
    if mirror_name == "official":
        return channel_name

    mirror_name = mirror_name.lower()
    if mirror_name not in CONDA_MIRRORS:
        raise ValueError(f"不支持的 conda 镜像: {mirror_name}")

    mirror_map = CONDA_MIRRORS[mirror_name]
    return mirror_map.get(channel_name, channel_name)


def get_pip_index_url(mirror_name: str) -> str:
    if mirror_name == "official":
        return ""

    mirror_name = mirror_name.lower()
    if mirror_name not in PIP_MIRRORS:
        raise ValueError(f"不支持的 pip 镜像: {mirror_name}")

    return PIP_MIRRORS[mirror_name]


def build_conda_channel_args(mirror: str, channel: str) -> List[str]:
    if mirror == "official":
        return ["--override-channels", "-c", channel]

    mirror_map = CONDA_MIRRORS[mirror]
    args = ["--override-channels"]

    primary = mirror_map.get(channel, channel)
    args.extend(["-c", primary])

    if "pkgs-main" in mirror_map:
        args.extend(["-c", mirror_map["pkgs-main"]])

    if platform.system().lower() == "windows" and "pkgs-msys2" in mirror_map:
        args.extend(["-c", mirror_map["pkgs-msys2"]])

    return args


# =========================
# 安装核心：一个包一个包安装
# =========================
def install_packages_individually(
    method: str,
    packages: List[str],
    cmd_builder,
    tracker: ProgressTracker,
    stage_title: str = "",
    is_installed=None,
) -> List[InstallResult]:
    results: List[InstallResult] = []
    if not packages:
        return results

    print_section(stage_title or f"{method} 单包安装阶段")
    tracker.show_stage(f"{stage_title or method}：共 {len(packages)} 个包，逐个安装")

    total = len(packages)
    for i, pkg in enumerate(packages, 1):
        bar = make_bar(i, total)
        pct = format_percent(i, total)
        print(color(f"\n{bar} {i}/{total} {pct} {method} 安装: {pkg}", Style.BOLD + Style.BLUE))

        if is_installed and is_installed(pkg):
            print_ok(f"{pkg} 已安装，跳过")
            results.append(InstallResult(pkg, method, True, message="skipped", duration_sec=0.0))
            tracker.advance(pkg, method, True, status_label="SKIP")
            continue

        cmd = cmd_builder(pkg)
        ok, output, sec = run_cmd_live(
            cmd,
            title=f"{method} install -> {pkg}",
            show_live=False,
            show_cmd=False,
        )

        if ok:
            print_ok(f"{pkg} 安装成功 ({sec:.1f}s)")
            results.append(InstallResult(pkg, method, True, duration_sec=sec))
            tracker.advance(pkg, method, True)
        else:
            print_err(f"{pkg} 安装失败 ({sec:.1f}s)")
            tail = tail_text(output, 2)
            if tail:
                print(color(tail, Style.DIM))
            results.append(
                InstallResult(pkg, method, False, message=output, duration_sec=sec)
            )
            tracker.advance(pkg, method, False)

    return results


def install_with_conda_groups(
    conda_exe: str,
    channel: str,
    packages: List[str],
    mirror: str,
    tracker: ProgressTracker,
    conda_installed: Optional[dict] = None,
) -> List[InstallResult]:
    if not packages:
        return []

    all_results: List[InstallResult] = []
    grouped = group_conda_packages(packages)
    channel_args = build_conda_channel_args(mirror, channel)
    method = get_conda_method_name(conda_exe)

    for group_name, group_pkgs in grouped:
        group_results = install_packages_individually(
            method=method,
            packages=group_pkgs,
            cmd_builder=lambda pkg: build_conda_install_cmd(conda_exe, channel_args, pkg),
            tracker=tracker,
            stage_title=f"{method} 分组安装：{group_name}",
            is_installed=(
                None
                if conda_installed is None
                else lambda req, _m=conda_installed: extract_package_name(req) in _m
            ),
        )
        all_results.extend(group_results)

    return all_results


def install_with_pip(
    packages: List[str],
    mirror: str,
    tracker: ProgressTracker,
    pip_installed: Optional[dict] = None,
) -> List[InstallResult]:
    if not packages:
        return []

    pip_index_url = get_pip_index_url(mirror)

    def build_pip_cmd(pkg: str) -> List[str]:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
        ]
        if pip_index_url:
            cmd.extend(["-i", pip_index_url])
        cmd.append(pkg)
        return cmd

    return install_packages_individually(
        method="pip",
        packages=packages,
        cmd_builder=build_pip_cmd,
        tracker=tracker,
        stage_title="pip 单包安装",
        is_installed=(
            None
            if pip_installed is None
            else lambda req, _m=pip_installed: extract_package_name(req) in _m
        ),
    )


def filter_conda_fallback_packages(failed_pip: List[InstallResult]) -> List[str]:
    pkgs: List[str] = []
    for item in failed_pip:
        req = item.name
        if "://" in req or req.startswith("git+"):
            continue
        m = REQ_LINE_RE.match(req)
        if not m:
            continue
        raw_name = m.group(1)
        norm = normalize_name(raw_name)
        if norm in PIP_ONLY_PACKAGES:
            continue
        conda_name = CONDA_NAME_MAP.get(norm, norm)
        version_part = req[len(raw_name):].strip()
        conda_req = f"{conda_name}{version_part}" if version_part else conda_name
        pkgs.append(conda_req)
    return sort_versioned_first(pkgs)


# =========================
# 展示
# =========================
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
            label = color("OK", Style.GREEN) if item.success else color("FAIL", Style.RED)
            print(f"  [{label}] {item.name} ({item.duration_sec:.1f}s)")
            if (not item.success) and item.message:
                tail = tail_text(item.message, 8)
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


def print_plan(conda_pkgs: List[str], pip_pkgs: List[str], conda_exe: str, channel: str, mirror: str):
    print_header("安装计划预览")
    print_info(f"操作系统: {platform.system()} {platform.release()}")
    print_info(f"Python: {sys.version.split()[0]}")
    print_info(f"当前环境: {os.environ.get('CONDA_DEFAULT_ENV', '(unknown)')}")
    print_info(f"环境路径: {os.environ.get('CONDA_PREFIX', '(unknown)')}")
    print_info(f"Conda执行器: {conda_exe}")
    print_info(f"Conda频道: {channel}")
    print_info(f"镜像源: {mirror}")
    print_info(f"Conda实际地址: {get_conda_channel_url(mirror, channel)}")
    pip_url = get_pip_index_url(mirror)
    print_info(f"pip实际地址: {pip_url if pip_url else '默认官方源'}")

    print_section(f"Conda/Mamba 优先安装 ({len(conda_pkgs)} 个)")
    for group_name, group_pkgs in group_conda_packages(conda_pkgs):
        print(color(f"  [{group_name}] ({len(group_pkgs)} 个)", Style.BOLD))
        for pkg in group_pkgs:
            print(f"    - {pkg}")

    print_section(f"pip 安装 ({len(pip_pkgs)} 个)")
    for pkg in pip_pkgs:
        print(f"  - {pkg}")


# =========================
# 主入口
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="在已激活 conda 环境中，按 conda/pip 分流，并逐个包安装 requirements。"
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
        "--mirror",
        default="tsinghua",
        choices=["official", "tsinghua", "ustc"],
        help="镜像源，默认 tsinghua",
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

        print_section("后台检查已安装包")
        conda_installed = get_conda_installed_map(conda_exe)
        pip_installed = get_pip_installed_map()
        print_info(f"Conda 已安装包: {len(conda_installed)} 个")
        print_info(f"pip 已安装包: {len(pip_installed)} 个")

        print_plan(conda_pkgs, pip_pkgs, conda_exe, args.channel, args.mirror)

        if args.dry_run:
            print_warn("dry-run 模式：未执行安装。")
            return

        summary = Summary()
        tracker = ProgressTracker(total_packages=len(conda_pkgs) + len(pip_pkgs))

        conda_results = install_with_conda_groups(
            conda_exe, args.channel, conda_pkgs, args.mirror, tracker, conda_installed
        )
        for r in conda_results:
            if r.success:
                summary.conda_success.append(r)
            else:
                summary.conda_failed.append(r)

        pip_results = install_with_pip(
            pip_pkgs, args.mirror, tracker, pip_installed
        )
        for r in pip_results:
            if r.success:
                summary.pip_success.append(r)
            else:
                summary.pip_failed.append(r)

        if summary.pip_failed:
            fallback_pkgs = filter_conda_fallback_packages(summary.pip_failed)
            if fallback_pkgs:
                print_section(f"pip 失败后改用 mamba/conda 安装 ({len(fallback_pkgs)} 个)")
                fallback_results = install_with_conda_groups(
                    conda_exe, args.channel, fallback_pkgs, args.mirror, tracker
                )

                fallback_success_names = {
                    extract_package_name(r.name) for r in fallback_results if r.success
                }
                if fallback_success_names:
                    summary.pip_failed = [
                        r for r in summary.pip_failed
                        if extract_package_name(r.name) not in fallback_success_names
                    ]

                for r in fallback_results:
                    if r.success:
                        summary.conda_success.append(r)
                    else:
                        summary.conda_failed.append(r)

        print_summary(summary)

        if summary.conda_failed or summary.pip_failed:
            sys.exit(1)

    except Exception as e:
        print_err(str(e))
        sys.exit(2)


if __name__ == "__main__":
    main()