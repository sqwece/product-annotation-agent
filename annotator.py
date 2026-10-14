#!/usr/bin/env python3
"""
商品识别 + 智能数据标注 Agent
Product Recognition & Agent-Assisted Annotation
================================================

一句话价值：
    用 Agent（LLM）自动完成商品分类标注，仅对低置信度样本做人工复核，
    把「全量人工标注」变成「机器标注 + 少量人工兜底」，从而显著降低人力成本。

标注闭环（Human-in-the-loop）：
    1. 向量召回   FAISS 检索 Top-K 候选类别（粗筛）
    2. Agent 标注 LLM 在候选类别中精排，给出类别 + 置信度 + 理由
    3. 置信度门控 高置信度 → 直接入库；低置信度 → 进入人工复核队列
    4. 人工复核   交互式确认 / 修正（只处理不确定样本，人力消耗最小化）
    5. 日志留痕   每次标注写入日志（文本日志 + 结构化审计记录）
    6. 数据集导出 导出最终标注结果，可直接用于下游训练 / 检索

技术栈：
    - 检索：FAISS 向量索引 + BGE 中文 embedding
    - 精排：DeepSeek LLM（OpenAI 兼容，经 langchain_openai）
    - 门控：置信度阈值 + 候选一致性校验（防 LLM 幻觉）

运行：
    python annotator.py                          # 自动标注 + 交互式人工复核
    python annotator.py --no-review              # 仅自动标注，跳过人工复核
    python annotator.py -p data/products.txt -c data/categories.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY
from embeddings import get_embeddings

# ========================== 路径与常量 ==========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "category_faiss_index"
OUTPUT_DIR = BASE_DIR / "output"

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
LLM_MODEL = "deepseek-chat"

CONFIDENCE_THRESHOLD = 0.6   # 置信度阈值：低于此值进入人工复核
TOP_K_RETRIEVE = 10          # FAISS 召回候选数（粗筛）
TOP_K_REFINE = 5             # 送入 LLM 精排的候选数


# ========================== 日志系统 ==========================

def setup_logging() -> logging.Logger:
    """同时输出到控制台与文件，实现标注审计留痕。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / "annotation.log"

    logger = logging.getLogger("annotator")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # 避免重复添加 handler

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()


# ========================== 数据结构 ==========================

@dataclass
class AnnotationResult:
    """单条商品的标注结果（含 Agent 标注与人工复核状态）。"""
    product: str
    category: str                 # 最终类别（Agent 标注或人工修正后）
    agent_category: str           # Agent 原始标注
    confidence: float             # Agent 置信度 0~1
    reason: str                   # 判断理由
    candidates: List[str] = field(default_factory=list)  # 召回候选类别
    source: str = "agent"         # 最终来源："agent" 或 "human"
    need_review: bool = False     # 是否需要人工复核
    reviewed: bool = False        # 是否已完成人工复核
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict:
        return asdict(self)


# ========================== 阶段1：离线构建类别向量库 ==========================

def build_or_load_category_index(categories_file: Path):
    """构建或加载类别 FAISS 索引；已存在则直接加载，否则从 categories.txt 构建并保存。"""
    if INDEX_DIR.exists():
        try:
            embedding_model = get_embeddings(EMBEDDING_MODEL)
            vector_store = FAISS.load_local(
                str(INDEX_DIR),
                embedding_model,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"已加载类别索引（{vector_store.index.ntotal} 个类别）")
            return vector_store
        except Exception as e:
            logger.warning(f"加载索引失败，将重新构建: {e}")

    categories = _read_lines(categories_file)
    logger.info(f"共 {len(categories)} 个候选类别，正在构建 FAISS 索引...")

    embedding_model = get_embeddings(EMBEDDING_MODEL)
    docs = [Document(page_content=cat, metadata={"category": cat}) for cat in categories]
    vector_store = FAISS.from_documents(docs, embedding_model)
    vector_store.save_local(str(INDEX_DIR))
    logger.info("类别索引构建完成并已保存。")
    return vector_store


def _read_lines(path: Path) -> List[str]:
    """读取文本文件，返回去空白、去空行的行列表。"""
    with open(path, encoding="utf-8") as f:
        return [x.strip() for x in f.readlines() if x.strip()]


# ========================== 阶段2：LLM 精排 ==========================

REFINE_PROMPT = ChatPromptTemplate.from_template(
    """
你是一个专业的超市商品分类 Agent。

现在有一个商品：
{product}

候选分类（仅限以下，请从中选择一个）：
{candidates}

你的任务：
根据商品真实用途、功能、使用场景，从候选分类中选择一个最符合的类别。

要求：
1. 必须从候选分类中选择。
2. 不允许创建新的分类。
3. 如果候选分类中没有合适的，返回"未知"。
4. 给出判断理由。

输出 JSON 格式：
{{"product":"", "category":"", "reason":"", "confidence":0-1}}
"""
)


def build_llm():
    """构建 DeepSeek LLM（OpenAI 兼容）。temperature=0 保证标注稳定可复现。"""
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0,
    )


# ========================== 阶段3：Agent 自动标注 ==========================

def annotate(product: str, retriever, chain) -> AnnotationResult:
    """两阶段标注：FAISS 粗筛召回候选 → LLM 精排 → 置信度门控 + 异常兜底。"""
    try:
        # ---- 阶段1：向量检索（粗筛）----
        retrieved_docs = retriever.invoke(product)
        if not retrieved_docs:
            logger.warning(f"商品 '{product}' 未召回任何候选类别")
            return _fallback(product, "未召回任何候选类别")

        candidate_categories = [doc.metadata["category"] for doc in retrieved_docs[:TOP_K_REFINE]]

        # ---- 阶段2：LLM 精排（Agent 标注）----
        raw = chain.invoke({
            "product": product,
            "candidates": "\n".join(candidate_categories),
        })

        # 字段完整性校验
        if not all(k in raw for k in ("product", "category", "reason", "confidence")):
            raise ValueError("LLM 返回缺少必要字段")

        category = raw["category"]
        reason = raw.get("reason", "")
        try:
            confidence = float(raw["confidence"])
        except (TypeError, ValueError):
            confidence = 0.0

        # ---- 候选一致性校验（防 LLM 幻觉）----
        if category not in candidate_categories and category != "未知":
            logger.warning(f"LLM 返回非候选类别 '{category}'，已修正为'未知'")
            reason = f"LLM 返回了非候选类别，已修正。原理由：{reason}"
            category = "未知"
            confidence = 0.3

        need_review = confidence < CONFIDENCE_THRESHOLD
        return AnnotationResult(
            product=product,
            category=category,
            agent_category=category,
            confidence=confidence,
            reason=reason,
            candidates=candidate_categories,
            source="agent",
            need_review=need_review,
            reviewed=False,
        )

    except Exception as e:
        logger.error(f"标注商品 '{product}' 时发生异常: {e}")
        return _fallback(product, f"处理异常: {e}")


def _fallback(product: str, reason: str) -> AnnotationResult:
    """兜底结果：标记为未知，进入人工复核。"""
    return AnnotationResult(
        product=product,
        category="未知",
        agent_category="未知",
        confidence=0.0,
        reason=reason,
        candidates=[],
        source="agent",
        need_review=True,
        reviewed=False,
    )


# ========================== 阶段4：人工复核 ==========================

def human_review(results: List[AnnotationResult], categories_full: List[str]) -> List[AnnotationResult]:
    """对低置信度样本进行交互式人工复核——只处理不确定项，人力消耗最小化。"""
    pending = [r for r in results if r.need_review]
    if not pending:
        logger.info("无低置信度样本，跳过人工复核")
        return results

    print(f"\n{'=' * 60}")
    print(f"🧑 人工复核：共 {len(pending)} 条低置信度样本需要确认")
    print(f"{'=' * 60}")

    for r in pending:
        print(f"\n商品：{r.product}")
        print(f"  Agent 标注：{r.agent_category}（置信度 {r.confidence:.2f}）")
        print(f"  理由：{r.reason}")
        print(f"  候选类别：{', '.join(r.candidates) if r.candidates else '（无召回）'}")

        while True:
            choice = input(
                f"  → 回车=接受 '{r.agent_category}'；"
                f"或输入正确类别（候选类别或全量类别，'未知'表示弃标）: "
            ).strip()

            if not choice:  # 接受 Agent 标注
                r.category = r.agent_category
                r.source, r.reviewed = "human", True
                logger.info(f"复核通过: {r.product} -> {r.category}（人工确认）")
                break
            if choice == "未知" or choice in categories_full:
                r.category = choice
                r.source, r.reviewed = "human", True
                action = "确认" if choice == r.agent_category else f"修正（原 {r.agent_category}）"
                logger.info(f"复核{action}: {r.product} -> {r.category}（人工）")
                break
            print("  ⚠️ 无效输入：请从候选/全量类别中选择，或输入 '未知'")

    return results


# ========================== 阶段5：导出与统计 ==========================

def export_dataset(results: List[AnnotationResult], audit_file: Path):
    """导出最终标注数据集（JSONL，可直接用于下游），并追加审计日志。"""
    dataset_file = OUTPUT_DIR / "annotated_dataset.jsonl"
    with open(dataset_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    with open(audit_file, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    logger.info(f"标注数据集已导出: {dataset_file}")
    logger.info(f"审计日志已追加: {audit_file}")


def print_report(results: List[AnnotationResult], elapsed: float):
    """打印标注效率报告，量化人力节省。"""
    total = len(results)
    auto_ok = sum(1 for r in results if not r.need_review)
    review_count = sum(1 for r in results if r.need_review)
    corrected = sum(1 for r in results if r.source == "human" and r.category != r.agent_category)

    print("\n" + "=" * 60)
    print("📊 标注效率报告")
    print("=" * 60)
    print(f"  总计: {total} 条")
    print(f"  Agent 自动通过（置信度 ≥ {CONFIDENCE_THRESHOLD}）: {auto_ok} 条")
    print(f"  需人工复核: {review_count} 条")
    print(f"  其中人工修正: {corrected} 条")
    print(f"  总耗时: {elapsed:.2f} 秒（平均 {elapsed / total:.2f} 秒/条）")

    # 对比纯人工（假设人工 30 秒/条）
    manual_time = total * 30
    speedup = manual_time / elapsed if elapsed > 0 else 0
    saved_ratio = (total - review_count) / total if total else 0
    print(f"  对比纯人工（30 秒/条）: 效率提升约 {speedup:.0f} 倍")
    print(f"  人工介入占比: {review_count / total * 100:.1f}%（即 {saved_ratio * 100:.0f}% 样本无需人工）")
    print("=" * 60)


# ========================== 主流程 ==========================

def run(products_file: Path, categories_file: Path, do_review: bool) -> List[AnnotationResult]:
    """完整标注流水线：建索引 → 自动标注 → 人工复核 → 导出 → 报告。"""
    categories_full = _read_lines(categories_file)

    vector_store = build_or_load_category_index(categories_file)
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})
    chain = REFINE_PROMPT | build_llm() | JsonOutputParser()

    products = _read_lines(products_file)
    logger.info(f"共 {len(products)} 个商品待标注 | 模式：FAISS 召回 Top-{TOP_K_RETRIEVE} → LLM 精排 Top-{TOP_K_REFINE}")

    start = datetime.now()
    results = [annotate(p, retriever, chain) for p in products]
    elapsed = (datetime.now() - start).total_seconds()

    if do_review:
        results = human_review(results, categories_full)
    else:
        logger.info("已跳过人工复核（--no-review）")

    audit_file = OUTPUT_DIR / "annotation_audit.jsonl"
    export_dataset(results, audit_file)
    print_report(results, elapsed)
    return results


def main():
    parser = argparse.ArgumentParser(description="商品识别 + 智能数据标注 Agent")
    parser.add_argument("-p", "--products", type=Path, default=DATA_DIR / "products.txt",
                        help="待标注商品列表文件")
    parser.add_argument("-c", "--categories", type=Path, default=DATA_DIR / "categories.txt",
                        help="候选类别文件")
    parser.add_argument("--no-review", action="store_true",
                        help="仅自动标注，跳过交互式人工复核")
    args = parser.parse_args()

    run(args.products, args.categories, do_review=not args.no_review)


if __name__ == "__main__":
    main()
