#!/usr/bin/env python3
"""
生成翻译结果汇总报告 JSON 文件

包含：
- 翻译统计（函数数、通过率等）
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def find_translation_stats(output_dir: Path) -> List[Dict[str, Any]]:
    """查找所有项目的翻译统计文件"""
    results = []
    
    for project_dir in output_dir.iterdir():
        if not project_dir.is_dir():
            continue
        
        # 提取项目名（去掉日期后缀）
        dir_name = project_dir.name
        parts = dir_name.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            project_name = parts[0]
            date_suffix = parts[1]
        else:
            project_name = dir_name
            date_suffix = ""
        
        # 查找 translation_stats.json
        stats_pattern = project_dir / "workspace" / "incremental_work" / "*" / "translate_by_*" / "translation_stats.json"
        stats_files = list(project_dir.glob("workspace/incremental_work/*/translate_by_*/translation_stats.json"))
        
        if stats_files:
            stats_file = stats_files[0]
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                
                # 计算通过率
                # 注意：compiled 已经是所有通过编译的函数数（包括直接通过和修复后通过）
                # repaired 是额外的统计，表示"经过修复才通过"的函数数
                total = stats.get('total', 0)
                compiled = stats.get('compiled', 0)  # 这已经是总通过数
                repaired = stats.get('repaired', 0)  # 其中经过修复的数量
                # 直接通过（一次编译成功）= 总通过 - 修复后通过
                direct_pass = compiled - repaired
                
                pass_rate = (compiled / total * 100) if total > 0 else 0
                
                # 查找编译错误日志
                compile_errors_file = stats_file.parent / "compile_errors.log"
                has_error_log = compile_errors_file.exists()
                
                results.append({
                    "project": project_name,
                    "date": date_suffix,
                    "total_functions": total,
                    "translated": stats.get('translated', 0),
                    "passed_total": compiled,  # 总通过数
                    "passed_directly": direct_pass,  # 一次通过（完美翻译）
                    "passed_after_repair": repaired,  # 修复后通过
                    "failed_reverted": stats.get('failed', 0),
                    "skipped": stats.get('skipped', 0),
                    "injection_failed": stats.get('injection_failed', 0),
                    "pass_rate_percent": round(pass_rate, 2),
                    "has_error_log": has_error_log,
                    "stats_file": str(stats_file.relative_to(output_dir))
                })
            except Exception as e:
                print(f"警告: 读取 {stats_file} 失败: {e}")
    
    return results

def generate_summary_report(output_dir: Path) -> Dict[str, Any]:
    """生成汇总报告
    
    Args:
        output_dir: 翻译输出目录
    """
    projects = find_translation_stats(output_dir)

    # 按通过率排序
    projects.sort(key=lambda x: x['pass_rate_percent'], reverse=True)
    
    # 计算总体统计
    total_functions = sum(p['total_functions'] for p in projects)
    total_passed = sum(p['passed_total'] for p in projects)
    total_direct = sum(p['passed_directly'] for p in projects)
    total_repaired = sum(p['passed_after_repair'] for p in projects)
    total_failed = sum(p['failed_reverted'] for p in projects)
    total_injection_failed = sum(p['injection_failed'] for p in projects)
    
    overall_pass_rate = (total_passed / total_functions * 100) if total_functions > 0 else 0
    
    # 完美翻译（一次编译通过，无需修复）
    perfect_rate = (total_direct / total_functions * 100) if total_functions > 0 else 0
    
    # 分类统计
    perfect_projects = [p for p in projects if p['pass_rate_percent'] == 100]
    high_pass_projects = [p for p in projects if 80 <= p['pass_rate_percent'] < 100]
    medium_pass_projects = [p for p in projects if 50 <= p['pass_rate_percent'] < 80]
    low_pass_projects = [p for p in projects if p['pass_rate_percent'] < 50]
    
    report = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "output_directory": str(output_dir),
            "total_projects": len(projects)
        },
        "overall_summary": {
            "total_functions": total_functions,
            "total_passed": total_passed,
            "passed_directly": total_direct,
            "passed_after_repair": total_repaired,
            "total_failed_reverted": total_failed,
            "total_injection_failed": total_injection_failed,
            "overall_pass_rate_percent": round(overall_pass_rate, 2),
            "direct_pass_rate_percent": round(perfect_rate, 2)
        },
        "project_categories": {
            "perfect_100_percent": {
                "count": len(perfect_projects),
                "projects": [p['project'] for p in perfect_projects]
            },
            "high_80_to_99_percent": {
                "count": len(high_pass_projects),
                "projects": [p['project'] for p in high_pass_projects]
            },
            "medium_50_to_79_percent": {
                "count": len(medium_pass_projects),
                "projects": [p['project'] for p in medium_pass_projects]
            },
            "low_below_50_percent": {
                "count": len(low_pass_projects),
                "projects": [p['project'] for p in low_pass_projects]
            }
        },
        "projects_detail": projects
    }
    
    return report

def main():
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="生成翻译结果汇总报告")
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "translation_outputs"),
        help="翻译输出目录"
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=None,
        help="汇总报告 JSON 输出路径（默认: <outputs-dir>/translation_summary_report.json）"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.outputs_dir)

    # 兼容 run 目录布局：translation_outputs/<run_dir>/{results,intermediate}/...
    projects_root = output_dir
    if (output_dir / "intermediate").is_dir():
        projects_root = output_dir / "intermediate"
    
    if not projects_root.exists():
        print(f"错误: 目录不存在: {projects_root}")
        return
    
    print("正在生成翻译结果汇总报告...")
    report = generate_summary_report(projects_root)
    
    # 保存 JSON 报告
    report_file = Path(args.report_out) if args.report_out else (output_dir / "translation_summary_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n报告已保存到: {report_file}")
    
    # 打印摘要
    summary = report['overall_summary']
    print("\n" + "=" * 60)
    print("翻译结果汇总")
    print("=" * 60)
    print(f"总项目数:           {report['meta']['total_projects']}")
    print(f"总函数数:           {summary['total_functions']}")
    print(f"通过函数数:         {summary['total_passed']}")
    print(f"  - 一次通过:       {summary['passed_directly']}")
    print(f"  - 修复后通过:     {summary['passed_after_repair']}")
    print(f"失败回退数:         {summary['total_failed_reverted']}")
    print(f"注入失败数:         {summary['total_injection_failed']}")
    print("-" * 60)
    print(f"总体通过率:         {summary['overall_pass_rate_percent']:.2f}%")
    print(f"一次通过率:         {summary['direct_pass_rate_percent']:.2f}%")
    print("-" * 60)
    
    cats = report['project_categories']
    print(f"100% 通过项目:      {cats['perfect_100_percent']['count']} 个")
    print(f"80-99% 通过项目:    {cats['high_80_to_99_percent']['count']} 个")
    print(f"50-79% 通过项目:    {cats['medium_50_to_79_percent']['count']} 个")
    print(f"<50% 通过项目:      {cats['low_below_50_percent']['count']} 个")
    print("=" * 60)


if __name__ == "__main__":
    main()
