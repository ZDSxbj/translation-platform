#!/usr/bin/env python3
"""
Jina Reranker 队列调度运行器

解决问题：
1. 多项目并行运行时 GPU 内存分配不均导致 OOM
2. 动态监控 GPU 内存，确保每次只在有足够内存时运行
3. 支持串行或并行模式（根据 GPU 数量自动选择）

使用方法：
    # 处理单个项目
    python run_jina_reranker_queued.py --project my_project
    
    # 处理多个项目（队列模式）
    python run_jina_reranker_queued.py --projects proj1,proj2,proj3
    
    # 从环境变量读取项目列表（适用于批量脚本）
    export RERANKER_PROJECTS="proj1,proj2,proj3"
    python run_jina_reranker_queued.py
    
    # 串行模式（一次只运行一个）
    python run_jina_reranker_queued.py --serial
"""

import os
import sys
import time
import argparse
import subprocess
import re
import threading
import queue
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime

# 尝试导入 torch（可选，用于部分环境的 CUDA 探测回退）
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ProjectTask:
    """项目任务"""
    project_name: str
    workspace_root: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    assigned_gpu: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None


class GPUMemoryMonitor:
    """GPU 内存监控"""
    
    def __init__(self):
        self.gpu_ids = self._get_gpu_ids()
        self.num_gpus = len(self.gpu_ids)
        self._lock = threading.Lock()
    
    def _query_nvidia_smi(self) -> tuple[list[int], dict[str, int]]:
        """Return ([gpu_ids], {uuid -> gpu_id}) from `nvidia-smi -L` (best-effort)."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return [], {}
            ids: list[int] = []
            uuid_map: dict[str, int] = {}
            for line in (result.stdout or "").splitlines():
                # Example: GPU 0: NVIDIA A100 ... (UUID: GPU-xxxx)
                m = re.match(r"^GPU\s+(?P<id>\d+):.*\(UUID:\s*(?P<uuid>[^)]+)\)\s*$", line.strip())
                if not m:
                    continue
                gpu_id = int(m.group("id"))
                ids.append(gpu_id)
                uuid_map[m.group("uuid")] = gpu_id
            return ids, uuid_map
        except Exception:
            return [], {}

    def _get_gpu_ids(self) -> list[int]:
        """Get visible GPU ids, honoring CUDA_VISIBLE_DEVICES when possible."""
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        smi_ids, uuid_map = self._query_nvidia_smi()

        if cvd is not None:
            raw = cvd.strip()
            if raw == "" or raw.lower() in ("none", "-1", "no"):
                return []
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if parts:
                # Common case: numeric indices.
                if all(p.isdigit() for p in parts):
                    return [int(p) for p in parts]
                # UUID case: map via nvidia-smi -L if possible.
                mapped: list[int] = []
                for p in parts:
                    if p in uuid_map:
                        mapped.append(uuid_map[p])
                    else:
                        # Unknown token; fall back to nvidia-smi ids.
                        mapped = []
                        break
                if mapped:
                    return mapped

        # No CUDA_VISIBLE_DEVICES restriction: prefer nvidia-smi ids.
        if smi_ids:
            return smi_ids

        # Fallback: torch device count (best-effort).
        if TORCH_AVAILABLE:
            try:
                n = torch.cuda.device_count()
                return list(range(n))
            except Exception:
                return []
        return []
    
    def get_free_memory_gb(self, gpu_id: int) -> float:
        """获取 GPU 空闲内存（GB）"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', f'--id={gpu_id}'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip()) / 1024  # MB -> GB
        except Exception:
            pass
        
        # 回退方案
        if TORCH_AVAILABLE:
            try:
                props = torch.cuda.get_device_properties(gpu_id)
                allocated = torch.cuda.memory_allocated(gpu_id)
                return (props.total_memory - allocated) / (1024**3)
            except Exception:
                pass
        
        return 0.0
    
    def get_all_gpu_status(self) -> Dict[int, Dict]:
        """获取所有 GPU 状态"""
        status = {}
        for gpu_id in self.gpu_ids:
            free_memory = self.get_free_memory_gb(gpu_id)
            status[gpu_id] = {
                "free_memory_gb": free_memory,
                "available": free_memory >= 8.0  # Jina Reranker 约需 8GB
            }
        return status
    
    def find_available_gpu(self, min_memory_gb: float = 8.0) -> Optional[int]:
        """找到一个有足够内存的 GPU"""
        best_gpu = None
        best_memory = 0.0
        
        for gpu_id in self.gpu_ids:
            free_memory = self.get_free_memory_gb(gpu_id)
            if free_memory >= min_memory_gb and free_memory > best_memory:
                best_gpu = gpu_id
                best_memory = free_memory
        
        return best_gpu
    
    def print_status(self):
        """打印 GPU 状态"""
        print("\n" + "=" * 50)
        print("GPU 状态")
        print("=" * 50)
        for gpu_id in self.gpu_ids:
            free_memory = self.get_free_memory_gb(gpu_id)
            available = "✓" if free_memory >= 8.0 else "✗"
            print(f"  GPU {gpu_id}: {free_memory:.1f} GB 空闲 [{available}]")
        print("=" * 50 + "\n")


class JinaRerankerRunner:
    """Jina Reranker 运行器"""
    
    def __init__(self, script_dir: Path, serial_mode: bool = False):
        self.script_dir = script_dir
        self.serial_mode = serial_mode
        self.gpu_monitor = GPUMemoryMonitor()
        self.task_queue: queue.Queue = queue.Queue()
        self.results: List[ProjectTask] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # 配置
        self.min_memory_gb = float(os.environ.get("JINA_MIN_MEMORY_GB", "8.0"))
        self.check_interval = float(os.environ.get("JINA_CHECK_INTERVAL", "10.0"))
        self.conda_env = os.environ.get("RERANKER_ENV_NAME", "jina_reranker_env")
    
    def add_project(self, project_name: str, workspace_root: Optional[str] = None):
        """添加项目到队列"""
        task = ProjectTask(project_name=project_name, workspace_root=workspace_root)
        self.task_queue.put(task)
    
    def _run_single_project(self, task: ProjectTask, gpu_id: int) -> bool:
        """运行单个项目的 Jina Reranker"""
        task.assigned_gpu = gpu_id
        task.start_time = time.time()
        task.status = "running"

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 开始: {task.project_name} (GPU {gpu_id})")

        try:
            # 构建环境变量
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PROJECT_NAME"] = task.project_name
            # GPU 已由队列调度器分配；下游 reranker 不应再做“全局 GPU 槽位/调度”（否则会把不同物理 GPU 错当成同一张卡）
            env["C2R_JINA_EXTERNAL_SCHEDULER"] = "1"

            # 传递工作空间路径
            if task.workspace_root:
                env["C2R_WORKSPACE_ROOT"] = task.workspace_root
            elif os.environ.get("C2R_WORKSPACE_ROOT"):
                env["C2R_WORKSPACE_ROOT"] = os.environ["C2R_WORKSPACE_ROOT"]

            # 打印调试信息
            print(f"  PROJECT_NAME: {task.project_name}")
            print(f"  CUDA_VISIBLE_DEVICES: {gpu_id}")
            print(f"  C2R_WORKSPACE_ROOT: {env.get('C2R_WORKSPACE_ROOT', 'NOT SET')}")

            # 构建运行命令（直接运行Python，不使用conda run）
            cmd = [sys.executable, str(self.script_dir / "resort_by_unixcoder.py")]
            
            # 运行（实时输出，不捕获）
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 运行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=False,  # 改为 False，允许实时输出
                text=True,
                timeout=3600  # 1 小时超时
            )
            
            task.end_time = time.time()
            duration = task.end_time - task.start_time
            
            if result.returncode == 0:
                task.status = "completed"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 完成: {task.project_name} ({duration:.1f}s)")
                return True
            else:
                task.status = "failed"
                task.error = result.stderr[-500:] if result.stderr else "Unknown error"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 失败: {task.project_name}")
                if task.error:
                    print(f"  错误: {task.error[:200]}...")
                return False
                
        except subprocess.TimeoutExpired:
            task.status = "failed"
            task.error = "超时 (>1小时)"
            task.end_time = time.time()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 超时: {task.project_name}")
            return False
            
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.end_time = time.time()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 异常: {task.project_name} - {e}")
            return False
        
        finally:
            with self._lock:
                self.results.append(task)
    
    def _worker_loop(self, worker_id: int, gpu_id: int):
        """工作线程（固定绑定到一个 GPU，避免多个 worker 同时选择 GPU0）"""
        print(f"[Worker {worker_id}] 启动 (GPU {gpu_id})")

        while not self._stop_event.is_set():
            try:
                # 尝试获取任务
                try:
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if task is None:  # 停止信号
                    self.task_queue.task_done()  # 标记停止信号完成
                    break

                # 等待该 worker 绑定的 GPU 满足内存阈值
                wait_logged = False
                while not self._stop_event.is_set():
                    free_memory = self.gpu_monitor.get_free_memory_gb(gpu_id)
                    if free_memory >= self.min_memory_gb:
                        break
                    if not wait_logged:
                        print(f"[Worker {worker_id}] GPU {gpu_id} 内存不足，等待释放...")
                        self.gpu_monitor.print_status()
                        wait_logged = True
                    time.sleep(self.check_interval)

                if self._stop_event.is_set():
                    self.task_queue.task_done()  # 标记当前任务完成
                    self.task_queue.put(task)  # 放回队列供下次处理
                    break

                # 执行任务
                try:
                    self._run_single_project(task, gpu_id)
                finally:
                    # 标记任务完成（必须调用，否则 join() 会永远阻塞）
                    self.task_queue.task_done()

                # 任务完成后等待一下，让 GPU 内存释放
                time.sleep(2.0)

            except Exception as e:
                print(f"[Worker {worker_id}] 错误: {e}")
                import traceback
                traceback.print_exc()
                # 异常情况下也要标记任务完成，避免 join() 永远阻塞
                try:
                    self.task_queue.task_done()
                except ValueError:
                    pass  # 可能已经调用过了

        print(f"[Worker {worker_id}] 停止 (GPU {gpu_id})")
    
    def run_serial(self):
        """串行模式运行"""
        print("使用串行模式...")
        self.gpu_monitor.print_status()
        
        total = self.task_queue.qsize()
        completed = 0
        
        while not self.task_queue.empty():
            task = self.task_queue.get()
            
            # 等待 GPU
            gpu_id = None
            while gpu_id is None:
                gpu_id = self.gpu_monitor.find_available_gpu(self.min_memory_gb)
                if gpu_id is None:
                    print("等待 GPU 内存释放...")
                    self.gpu_monitor.print_status()
                    time.sleep(self.check_interval)
            
            # 运行
            self._run_single_project(task, gpu_id)
            completed += 1
            print(f"进度: {completed}/{total}")
            
            # 等待 GPU 内存释放
            time.sleep(5.0)

    def run_parallel(self, num_workers: int = None):
        """并行模式运行"""
        if num_workers is None:
            # 默认 workers 数 = GPU 数量（使用全部 GPU）
            num_workers = max(1, self.gpu_monitor.num_gpus)
        else:
            # Clamp to a sensible range (>=1, <= visible GPU count when known)
            try:
                num_workers = int(num_workers)
            except Exception:
                num_workers = max(1, self.gpu_monitor.num_gpus)
            num_workers = max(1, num_workers)
            if self.gpu_monitor.num_gpus > 0:
                num_workers = min(num_workers, self.gpu_monitor.num_gpus)

        print(f"使用并行模式 ({num_workers} workers)...")
        self.gpu_monitor.print_status()
        
        total = self.task_queue.qsize()
        
        # 启动工作线程：每个 worker 固定绑定到一个 GPU
        workers = []
        for i in range(num_workers):
            gpu_id = self.gpu_monitor.gpu_ids[i]
            t = threading.Thread(target=self._worker_loop, args=(i, gpu_id), daemon=True)
            t.start()
            workers.append(t)
        
        # 等待所有任务完成
        self.task_queue.join()
        
        # 发送停止信号
        for _ in range(num_workers):
            self.task_queue.put(None)
        
        # 等待线程结束
        for t in workers:
            t.join(timeout=10)
    
    def run(self, num_workers: int = None):
        """运行所有任务"""
        if self.gpu_monitor.num_gpus <= 0:
            print("错误: 未检测到可用 GPU（CUDA_VISIBLE_DEVICES / nvidia-smi / torch.cuda 都不可用）。")
            self.gpu_monitor.print_status()
            return
        if self.serial_mode or self.gpu_monitor.num_gpus <= 1 or (num_workers is not None and int(num_workers) <= 1):
            self.run_serial()
        else:
            self.run_parallel(num_workers=num_workers)
    
    def print_summary(self):
        """打印结果摘要"""
        print("\n" + "=" * 60)
        print("执行结果摘要")
        print("=" * 60)
        
        completed = [r for r in self.results if r.status == "completed"]
        failed = [r for r in self.results if r.status == "failed"]
        
        print(f"总计: {len(self.results)}, 成功: {len(completed)}, 失败: {len(failed)}")
        
        if completed:
            print("\n成功的项目:")
            for r in completed:
                duration = (r.end_time - r.start_time) if r.end_time and r.start_time else 0
                print(f"  ✓ {r.project_name} ({duration:.1f}s, GPU {r.assigned_gpu})")
        
        if failed:
            print("\n失败的项目:")
            for r in failed:
                print(f"  ✗ {r.project_name}: {r.error[:100] if r.error else 'Unknown'}")
        
        print("=" * 60)
        
        return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(description="Jina Reranker 队列调度运行器")
    parser.add_argument("--project", type=str, help="单个项目名称")
    parser.add_argument("--projects", type=str, help="项目列表（逗号分隔）")
    parser.add_argument("--workspace", type=str, help="工作空间根目录（单项目时使用）")
    parser.add_argument("--intermediate-dir", type=str, help="中间目录基础路径（多项目时使用，自动构造 <intermediate-dir>/<project>/workspace）")
    parser.add_argument("--serial", action="store_true", help="强制使用串行模式")
    parser.add_argument("--workers", type=int, default=None, help="并行 worker 数（默认=可见 GPU 数）")

    args = parser.parse_args()

    # 获取脚本目录
    script_dir = Path(__file__).parent.resolve()

    # 确定项目列表
    projects = []
    
    if args.project:
        projects = [args.project]
    elif args.projects:
        projects = [p.strip() for p in args.projects.split(",") if p.strip()]
    elif os.environ.get("RERANKER_PROJECTS"):
        projects = [p.strip() for p in os.environ["RERANKER_PROJECTS"].split(",") if p.strip()]
    elif os.environ.get("PROJECT_NAME"):
        projects = [os.environ["PROJECT_NAME"]]
    
    if not projects:
        print("错误: 未指定项目。使用 --project, --projects 或设置 RERANKER_PROJECTS 环境变量")
        return 1
    
    print(f"准备处理 {len(projects)} 个项目: {', '.join(projects[:5])}{'...' if len(projects) > 5 else ''}")
    
    # 创建运行器
    runner = JinaRerankerRunner(script_dir, serial_mode=args.serial)
    
    # 添加任务
    for project in projects:
        # 确定工作空间路径
        workspace = None
        if args.intermediate_dir:
            # 使用中间目录构造路径：<intermediate-dir>/<project>/workspace
            workspace = str(Path(args.intermediate_dir) / project / "workspace")
        elif args.workspace:
            workspace = args.workspace
        else:
            # 尝试从环境变量获取
            workspace = os.environ.get("C2R_WORKSPACE_ROOT")

        if workspace:
            print(f"  项目 {project}: 工作空间 = {workspace}")
        runner.add_project(project, workspace)
    
    # 运行
    workers = args.workers
    if workers is None:
        env_workers = os.environ.get("JINA_WORKERS", "").strip()
        if env_workers.isdigit():
            try:
                workers = int(env_workers)
            except Exception:
                workers = None
    runner.run(num_workers=workers)
    
    # 打印摘要
    success = runner.print_summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
