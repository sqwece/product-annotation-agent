#!/usr/bin/env python3
"""
标注方案评测（Benchmark）：速度 + 准度
=====================================

对带 ground-truth 的评测集运行「两阶段标注」（FAISS 粗筛 → LLM 精排 → 置信度门控），
统计：
  1. 速度：总耗时、平均每条耗时、最快/最慢
  2. 准度：准确率（最终类别 vs 人工标签）、按类别的命中情况
  3. 低置信度样本：列出低于阈值的样本，附 Top-10 候选 + 相似度（供人工复核参考）

运行：
    python eval.py -e data/eval_set.json -c data/categories.txt
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from langchain_core.output_parsers import JsonOutputParser

from annotator import (
    REFINE_PROMPT,
    AnnotationResult,
    build_llm,
    build_or_load_category_index,
    annotate,
    logger,
)

# Windows 控制台默认 GBK，强制 UTF-8 输出，避免中文/emoji 打印报错
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_EVAL_SET = Path(__file__).resolve().parent / "data" / "eval_set.json"
DEFAULT_CATEGORIES = Path(__file__).resolve().parent / "data" / "categories.txt"


def load_eval_set(path: Path) -> List[dict]:
    """加载评测集：[{"product": "...", "label": "..."}]"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate(eval_set: List[dict], categories_file: Path) -> dict:
    """运行两阶段标注并统计速度、准度、低置信度样本。"""
    vector_store = build_or_load_category_index(categories_file)
    chain = REFINE_PROMPT | build_llm() | JsonOutputParser()

    timings: List[float] = []
    results: List[AnnotationResult] = []
    low_conf: List[Tuple[dict, AnnotationResult]] = []

    start_total = datetime.now()
    for item in eval_set:
        t0 = datetime.now()
        r = annotate(item["product"], vector_store, chain)
        timings.append((datetime.now() - t0).total_seconds())
        results.append(r)
        if r.need_review:
            low_conf.append((item, r))
    total_elapsed = (datetime.now() - start_total).total_seconds()

    correct = sum(1 for item, r in zip(eval_set, results) if r.category == item["label"])
    total = len(eval_set)

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "total_elapsed": total_elapsed,
        "avg_per_item": total_elapsed / total if total else 0.0,
        "min_per_item": min(timings) if timings else 0.0,
        "max_per_item": max(timings) if timings else 0.0,
        "p50_per_item": statistics.median(timings) if timings else 0.0,
        "low_conf": low_conf,
        "results": results,
    }


def print_benchmark(metrics: dict, eval_set: List[dict]):
    """打印评测报告。"""
    total = metrics["total"]
    print("\n" + "=" * 64)
    print("📊 两阶段标注方案评测报告")
    print("=" * 64)

    # 速度
    print("\n【速度】")
    print(f"  总耗时:        {metrics['total_elapsed']:.2f} 秒")
    print(f"  平均每条:      {metrics['avg_per_item']:.2f} 秒")
    print(f"  最快 / 最慢:   {metrics['min_per_item']:.2f}s / {metrics['max_per_item']:.2f}s")
    print(f"  中位数:        {metrics['p50_per_item']:.2f} 秒/条")
    manual_time = total * 30
    speedup = manual_time / metrics["total_elapsed"] if metrics["total_elapsed"] > 0 else 0
    print(f"  对比纯人工(30s/条): 提升约 {speedup:.1f} 倍")

    # 准度
    print("\n【准度】")
    print(f"  正确: {metrics['correct']} / {total}")
    print(f"  准确率: {metrics['accuracy'] * 100:.1f}%")

    # 逐条明细
    print("\n【逐条结果】")
    for item, r in zip(eval_set, metrics["results"]):
        mark = "✅" if r.category == item["label"] else "❌"
        conf = f"{r.confidence:.2f}" if r.confidence >= 0 else "—"
        print(f"  {mark} {r.product:<8s} 标注={r.category:<5s} 标签={item['label']:<5s} 置信度={conf}")

    # 低置信度样本（附 Top-10）
    low_conf = metrics["low_conf"]
    print("\n【低置信度样本（需人工复核，附 Top-10 候选）】")
    if not low_conf:
        print("  （无）")
    for item, r in low_conf:
        print(f"\n  ▸ 商品：{r.product}（正确标签：{item['label']}）")
        print(f"    Agent 标注：{r.agent_category}（置信度 {r.confidence:.2f}）")
        print(f"    理由：{r.reason}")
        if r.top10_candidates:
            print(f"    Top-{len(r.top10_candidates)} 候选：")
            for i, c in enumerate(r.top10_candidates, 1):
                marker = " ← 送入 LLM" if c["category"] in r.candidates else ""
                print(f"      {i:2d}. {c['category']:<8s} 相似度 {c['score']:.4f}{marker}")
        else:
            print("    Top-10 候选：（无召回）")
    print("\n" + "=" * 64)


def main():
    parser = argparse.ArgumentParser(description="两阶段标注方案评测（速度 + 准度）")
    parser.add_argument("-e", "--eval-set", type=Path, default=DEFAULT_EVAL_SET,
                        help="评测集 JSON 文件（product + label）")
    parser.add_argument("-c", "--categories", type=Path, default=DEFAULT_CATEGORIES,
                        help="候选类别文件")
    args = parser.parse_args()

    eval_set = load_eval_set(args.eval_set)
    logger.info(f"开始评测：{len(eval_set)} 条样本")

    metrics = evaluate(eval_set, args.categories)

    # 保存完整结果
    out = Path(__file__).resolve().parent / "output" / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: ([(i, r.to_dict()) for i, r in v] if k == "low_conf" else
            [r.to_dict() for r in v] if k == "results" else v)
        for k, v in metrics.items()
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    logger.info(f"评测结果已保存: {out}")

    print_benchmark(metrics, eval_set)


if __name__ == "__main__":
    main()
