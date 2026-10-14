# 商品识别 + 智能数据标注 Agent

用 **Agent（LLM）自动完成商品分类标注**，仅对低置信度样本做人工复核，把「全量人工标注」变成「机器标注 + 少量人工兜底」，**显著降低标注人力成本**。

## 核心价值

| 传统人工标注 | 本方案（Agent 介入） |
|---|---|
| 每条样本都靠人看、人标 | Agent 自动标注，**高置信度直接入库** |
| 人力成本随数据量线性增长 | 只有**低置信度样本**才进入人工复核队列 |
| 无留痕，难追溯 | 每次标注**写入日志 + 审计记录**，可追溯 |
| 标注结果散落 | 一键**导出结构化标注数据集**（JSONL） |

## 标注闭环（Human-in-the-loop）

```
┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌────────────┐   ┌──────────────┐
│ 1. 向量召回  │ → │ 2. Agent 标注 │ → │ 3. 置信度门控 │ → │ 4. 人工复核 │ → │ 5. 日志留痕   │
│ FAISS Top-K │   │ LLM 精排     │   │ 高→入库      │   │ 低→交互修正 │   │ 6. 数据集导出 │
│ （粗筛）     │   │ 类别+置信度   │   │ 低→复核队列  │   │ （仅不确定项）│   │ （JSONL）    │
└─────────────┘   └──────────────┘   └─────────────┘   └────────────┘   └──────────────┘
```

1. **向量召回**：FAISS 检索 Top-K 候选类别（BGE 中文 embedding）
2. **Agent 标注**：LLM 在候选类别中精排，输出类别 + 置信度 + 理由
3. **置信度门控**：置信度 ≥ 阈值 → 直接入库；否则进入人工复核队列
4. **人工复核**：交互式确认/修正，**只处理不确定样本**
5. **日志留痕**：文本日志 `annotation.log` + 结构化审计 `annotation_audit.jsonl`
6. **数据集导出**：`annotated_dataset.jsonl`，可直接用于下游训练/检索

## 工程亮点

- **两阶段检索**：FAISS 粗筛 + LLM 精排，兼顾召回与精度
- **防幻觉校验**：LLM 返回非候选类别时自动修正为「未知」并降置信度
- **异常兜底**：召回为空 / LLM 异常 / 字段缺失均有兜底，不中断流水线
- **审计留痕**：每条标注带时间戳，Agent 标注 vs 人工修正的 `source` 字段可追溯
- **效率量化**：自动通过率、人力介入占比、对比纯人工的效率提升倍数

## 技术栈

- **检索**：FAISS + `sentence-transformers`（BGE-small-zh 中文 embedding）
- **精排**：DeepSeek LLM（OpenAI 兼容，经 `langchain_openai`）
- **门控**：置信度阈值 + 候选一致性校验

## 项目结构

```
product-annotation-agent/
├── annotator.py       # 主程序（标注闭环全流程）
├── eval.py            # 评测脚本（速度 + 准度）
├── embeddings.py      # BGE embedding 封装（首次自动下载，后续离线）
├── config.py          # 从 .env 读取 API Key（含 DeepSeek 直连配置）
├── data/
│   ├── categories.txt      # 候选类别库
│   ├── products.txt        # 待标注商品列表
│   ├── eval_set.json       # 评测集（26 条常规样本，带 ground-truth）
│   └── eval_set_hard.json  # 困难评测集（含无法归类的样本）
└── requirements.txt
```

## 快速开始

1. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

2. **配置密钥**：在项目根目录建 `.env`（不入库）

   ```bash
   OPENAI_API_KEY=sk-...   # DeepSeek 的 key
   ```

3. **运行**

   ```bash
   python annotator.py                        # 自动标注 + 交互式人工复核
   python annotator.py --no-review            # 仅自动标注，跳过人工复核
   python annotator.py -p data/products.txt -c data/categories.txt
   ```

   首次运行会自动下载 BGE 嵌入模型到 `models/`（需能访问 Hugging Face），之后可离线运行。

## 评测（速度 + 准度）

```bash
python eval.py -e data/eval_set.json        # 常规样本（26 条）
python eval.py -e data/eval_set_hard.json   # 困难样本（含无法归类）
```

参考结果（本机实测，受网络延迟影响）：

| 指标 | 常规集 | 困难集 |
|---|---|---|
| 样本数 | 26 | 8 |
| 准确率 | 100%（26/26） | 62.5%（5/8） |
| 平均耗时 | 1.01 秒/条 | 1.04 秒/条 |
| 对比纯人工(30s/条) | ~30 倍 | ~29 倍 |

低置信度样本会在报告末尾列出，**附 Top-N 候选及相似度**（检索 `Top-10`，当类别库不足 10 类时显示全部类别）。

> 已知局限：困难集的主要误差来自「LLM 强行归类」——对无法匹配的商品（如充电宝、宠物粮）没有返回"未知"，而是塞进最接近的类别。可通过收紧 prompt、增加"未知"示例、或引入类别覆盖度二次校验来缓解。

## 输出说明

运行后 `output/` 目录会生成：

| 文件 | 说明 |
|---|---|
| `annotated_dataset.jsonl` | 最终标注数据集（每条一行 JSON，含类别/置信度/来源/时间戳） |
| `annotation_audit.jsonl`  | 追加式审计日志（每次运行累加，完整追溯） |
| `annotation.log`          | 文本日志（控制台 + 文件双输出） |

## 免责声明

本项目为 Agent 编排与「人机协同标注」的实践示例，非生产级标注平台。
