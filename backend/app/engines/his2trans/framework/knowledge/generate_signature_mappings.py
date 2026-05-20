#!/usr/bin/env python3
"""
生成函数签名映射文件

在阶段1完成后运行，生成：
1. rust_signature_by_uid.json: uid -> rust_signature
2. func_file_to_rust_sig.json: func_file -> rust_signature

这避免了阶段2/3需要用BM25猜测C函数到Rust签名的对应关系。

使用方法：
    python generate_signature_mappings.py --project <project_name>
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from project_config import PROJECT_NAME, PROJECT_ROOT
from workspace_config import get_workspace_root
from call_graph import generate_function_uid


def load_manifest(project_name: str) -> Dict:
    """加载 functions_manifest.json"""
    workspace_root = get_workspace_root()
    manifest_path = workspace_root / "extracted" / project_name / "functions_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"函数清单不存在: {manifest_path}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_function_signatures(project_name: str) -> Dict:
    """加载 function_signatures.json"""
    workspace_root = get_workspace_root()
    sig_path = workspace_root / "source_skeletons" / project_name / "function_signatures.json"

    if not sig_path.exists():
        raise FileNotFoundError(f"骨架签名文件不存在: {sig_path}\n请先运行骨架生成！")

    with open(sig_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_uid_to_signature_mapping(manifest: Dict, signatures: Dict) -> Dict:
    """
    生成 uid -> rust_signature 映射

    Args:
        manifest: functions_manifest.json 的内容
        signatures: function_signatures.json 的内容

    Returns:
        uid -> rust_signature 字典
    """
    uid_to_sig = {}

    # function_signatures.json 的格式: {name: {"c_signature": ..., "rust_signature": ..., "source_file": ...}}
    # 为每个签名生成 uid
    for name, sig_info in signatures.items():
        source_file = sig_info.get('source_file', '')
        # 从 manifest 中查找匹配的函数以获取行号
        # 通过名称和源文件匹配
        for func_meta in manifest['functions']:
            if func_meta['name'] == name and func_meta['source_file'] == source_file:
                uid = func_meta['uid']
                uid_to_sig[uid] = sig_info.get('rust_signature', '')
                break
        else:
            # 如果在 manifest 中找不到，尝试仅通过名称匹配（可能有多个同名函数）
            # 这种情况下，使用第一个匹配项
            for func_meta in manifest['functions']:
                if func_meta['name'] == name:
                    uid = func_meta['uid']
                    uid_to_sig[uid] = sig_info.get('rust_signature', '')
                    break

    return uid_to_sig


def generate_funcfile_to_signature_mapping(manifest: Dict, uid_to_sig: Dict) -> Dict:
    """
    生成 func_file -> rust_signature 映射

    Args:
        manifest: functions_manifest.json 的内容
        uid_to_sig: uid -> rust_signature 映射

    Returns:
        func_file -> rust_signature 字典
    """
    funcfile_to_sig = {}

    for func_meta in manifest['functions']:
        func_file = func_meta['func_file']
        uid = func_meta['uid']
        rust_sig = uid_to_sig.get(uid)

        if rust_sig:
            funcfile_to_sig[func_file] = rust_sig
        else:
            # 如果没有找到 Rust 签名，记录为 None
            funcfile_to_sig[func_file] = None

    return funcfile_to_sig


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='生成函数签名映射文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
    python generate_signature_mappings.py --project approach_ble
        """
    )
    parser.add_argument('--project', type=str, default=PROJECT_NAME, help='项目名称')
    args = parser.parse_args()

    project_name = args.project
    workspace_root = get_workspace_root()
    output_dir = workspace_root / "source_skeletons" / project_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"生成函数签名映射文件")
    print(f"{'='*60}")
    print(f"项目: {project_name}")
    print(f"输出目录: {output_dir}\n")

    # 1. 加载函数清单
    print("步骤 1/4: 加载函数清单...")
    try:
        manifest = load_manifest(project_name)
        print(f"  ✓ 加载了 {manifest['total_functions']} 个函数的元数据")
    except FileNotFoundError as e:
        print(f"  ✗ 错误: {e}")
        print("\n提示: 请先运行 get_dependencies.py 生成函数清单")
        return 1

    # 2. 加载骨架签名
    print("\n步骤 2/4: 加载骨架签名...")
    try:
        signatures = load_function_signatures(project_name)
        print(f"  ✓ 加载了 {len(signatures)} 个函数的 Rust 签名")
    except FileNotFoundError as e:
        print(f"  ✗ 错误: {e}")
        return 1

    # 3. 生成 uid -> rust_signature 映射
    print("\n步骤 3/4: 生成 uid -> rust_signature 映射...")
    uid_to_sig = generate_uid_to_signature_mapping(manifest, signatures)

    uid_to_sig_path = output_dir / "rust_signature_by_uid.json"
    with open(uid_to_sig_path, 'w', encoding='utf-8') as f:
        json.dump(uid_to_sig, f, indent=2, ensure_ascii=False)

    print(f"  ✓ 生成了 {len(uid_to_sig)} 个 uid 映射")
    print(f"  ✓ 已保存到: {uid_to_sig_path}")

    # 4. 生成 func_file -> rust_signature 映射
    print("\n步骤 4/4: 生成 func_file -> rust_signature 映射...")
    funcfile_to_sig = generate_funcfile_to_signature_mapping(manifest, uid_to_sig)

    funcfile_to_sig_path = output_dir / "func_file_to_rust_sig.json"
    with open(funcfile_to_sig_path, 'w', encoding='utf-8') as f:
        json.dump(funcfile_to_sig, f, indent=2, ensure_ascii=False)

    matched_count = sum(1 for sig in funcfile_to_sig.values() if sig is not None)
    print(f"  ✓ 生成了 {len(funcfile_to_sig)} 个 func_file 映射")
    print(f"  ✓ 成功匹配: {matched_count}/{len(funcfile_to_sig)}")
    print(f"  ✓ 已保存到: {funcfile_to_sig_path}")

    # 5. 汇总
    print(f"\n{'='*60}")
    print("📊 映射生成完成！")
    print(f"{'='*60}")
    print(f"  - rust_signature_by_uid.json: {len(uid_to_sig)} 个映射")
    print(f"  - func_file_to_rust_sig.json: {matched_count}/{len(funcfile_to_sig)} 个有效映射")
    print(f"\n提示：阶段3将优先使用 func_file_to_rust_sig.json 获取签名")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
