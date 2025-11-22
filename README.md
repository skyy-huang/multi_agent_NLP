# 多智能体学术写作优化系统 使用手册

> 双 Agent 协作 (优化 / 评审) + 轻量 ReAct 工具调用 + 向量记忆 (FAISS 回退) + 数据合成与评估 + 蒸馏与 LoRA/QLoRA 微调示例。
>
> 本项目用于快速验证“多智能体协作改写 + 教师信号蒸馏”在学术写作优化场景中的原型。

---
## 📚 目录
1. [项目简介](#项目简介)
2. [特性概览](#特性概览)
3. [核心概念](#核心概念)
4. [架构与流程图](#架构与流程图)
5. [环境与安装](#环境与安装)
6. [快速开始 Demo](#快速开始-demo)
7. [CLI 总览与参数详解](#cli-总览与参数详解)
8. [典型工作流示例](#典型工作流示例)
9. [数据格式规范](#数据格式规范)
10. [报告与可视化 (JSON / HTML)](#报告与可视化-json--html)
11. [评估指标说明](#评估指标说明)
12. [回退策略与健壮性](#回退策略与健壮性)
13. [环境变量一览](#环境变量一览)
14. [交互模式 (ENABLE_INTERACTIVE)](#交互模式-enable_interactive)
15. [性能与调优建议](#性能与调优建议)
16. [常见问题 FAQ](#常见问题-faq)
17. [扩展开发指南](#扩展开发指南)
18. [测试与验证](#测试与验证)
19. [Roadmap 后续扩展](#roadmap-后续扩展)
20. [许可与免责声明](#许可与免责声明)

---
## 项目简介
本项目演示一个用于“学术风格文本优化”的多智能体闭环：
- Agent A (Optimizer)：基于需求与记忆进行迭代改写；
- Agent B (Reviewer)：对 A 的输出结构化评分、指出问题、给出下一轮建议；
- 多轮协作收敛后产出最终优化文本，伴随完整轮次日志、评分、Diff、工具观察等信息；
- 支持批量合成数据 → 蒸馏 → 小模型 LoRA 微调的完整最小链路。

适用场景：原型验证 / Prompt 实验 / 数据生成 / 微调前置处理。当前为研究演示，尚未针对生产可靠性做深度加固。

---
## 特性概览
- 多轮协作：`--rounds` 控制迭代次数。
- 工具层：网络搜索 (SerpAPI)、Python REPL、文件读写。
- 记忆层：优先 FAISS + Embeddings；缺失依赖或 Key 时回退到简易字符串相似度检索。
- 健壮回退链：LLM / Embeddings / VectorStore / 搜索逐级降级，保证流程可运行。
- 数据合成：批量运行多案例写入 JSONL（含教师信号、评分、轮次日志）。
- 蒸馏：从合成数据抽取 `instruction/output` 训练对。
- 评估：自动计算多项轻量指标（长度、词汇多样性、重复度、可读性、连贯性、句长方差、二元组重复下降）。
- 报告：可输出结构化 JSON 与 HTML 可视化报告 (Diff 高亮)。
- LoRA / QLoRA：`lora_distill.py` 最小训练脚本。
- 可配置：语言 / 需求 / 种子文件 / 输出路径 / 是否禁用工具或记忆。

---
## 核心概念
| 概念 | 说明 |
|------|------|
| 轮次 (Round) | 一次优化 + 一次评审组成一个完整循环。|
| 教师信号 (teacher_signal) | 最后一轮优化后文本，作为蒸馏目标。|
| 蒸馏对 (distillation pair) | `{"instruction": ..., "output": ...}` 用于微调。|
| 需求 (requirements) | 用户期望提升的维度列表，如“学术表达提升, 结构清晰”。|
| 记忆检索 | 历史优化文本 / 反馈写入向量库并在后续轮次召回支撑一致性。|
| Diff | 前一轮与当前优化文本差异，统一 diff 语法高亮。|

---
## 架构与流程图
```
User Text + Requirements
          │
          ▼
     Agent A (Optimizer)  ←── 工具观察 (搜索 / REPL / 文件IO)
          │ 输出优化稿 + 修改说明
          ▼
     Agent B (Reviewer)
          │ 反馈 / 评分(JSON) / 下轮建议
          ▼
   MemoryManager (写入文本与反馈日志)
          │ 检索相似片段辅助下一轮
          └──► 多轮循环直到 rounds 完成
```
关键模块：
- `DualAgentAcademicSystem.collaborate()` 多轮主循环
- `_plan_and_act()` 条件触发工具调用
- `MemoryManager` 记忆添加与召回
- `synthesize_dataset()` 批量合成
- `evaluate()` 运行多案例指标统计
- `prepare_distillation_pairs()` 蒸馏数据生成

---
## 环境与安装
### 1. Python 版本
建议 Python 3.10+。

### 2. 创建虚拟环境 (Windows CMD)
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```
Linux / macOS:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### 3. 配置 `.env`
```
OPENAI_API_KEY=你的Key(可选)
SERPAPI_API_KEY=你的SerpAPI Key(可选)
OPENAI_BASE_URL=https://api.deepseek.com  # 或其它兼容OpenAI接口
LLM_MODEL=deepseek-reasoner               # 或 gpt-4o-mini / deepseek-chat
EMBED_MODEL_NAME=text-embedding-3-small   # 若 Base 支持 embeddings
ENABLE_INTERACTIVE=0                      # 设 1 开启交互模式
```
不配置 OPENAI_API_KEY 会自动使用 DummyLLM（占位输出，仅用于流程验证）。

---
## 快速开始 Demo
最小演示（中文，默认 2 轮）：
```bat
python multi_agent_nlp_project.py demo
```
自定义文本与需求：
```bat
python multi_agent_nlp_project.py demo --rounds 3 --text "这是一个测试初稿，结构略显松散，需要提升专业性。" --requirements "学术表达提升;逻辑结构优化"
```
英文模式：
```bat
python multi_agent_nlp_project.py demo --lang en --requirements "academic polish,logical coherence"
```
生成 JSON 报告与 HTML：
```bat
python multi_agent_nlp_project.py demo --report demo.json --html-report demo.html
```

---
## CLI 总览与参数详解
```text
python multi_agent_nlp_project.py [command] [--options]

command:
  demo        单条文本多轮优化
  synthesize  批量合成 JSONL 数据集
  eval        多案例评估指标计算
  distill     从合成 JSONL 生成蒸馏训练对

通用参数:
  --rounds <int>          协作轮次 (>=1)
  --requirements <str>    逗号/分号分隔需求列表
  --lang zh|en            语言 (影响默认初稿与默认需求)
  --no-tools              禁用工具调用 (消融实验)
  --no-memory             禁用向量记忆 (消融实验)
  --report <path>         输出 JSON 报告
  --html-report <path>    输出 HTML 报告 (demo/eval 支持)

demo 专用:
  --text <str>            初始文本

synthesize 专用:
  --seeds-file <path>     种子文本文件 (每行一条)，缺省使用内置 3 条
  --out <path>            输出 JSONL 文件路径

distill 专用:
  --distill-src <path>    上游合成 JSONL 数据源
  --distill-out <path>    蒸馏输出 JSONL
```

参数解析函数：`build_arg_parser()`（位于 `multi_agent_nlp_project.py`）。

---
## 典型工作流示例
### 1. 数据合成 → 蒸馏 → LoRA 微调
1. 合成原始多轮写作数据：
```bat
python multi_agent_nlp_project.py synthesize --rounds 3 --requirements "学术表达提升,结构清晰,可读性增强" --out data/synth.jsonl
```
2. 生成蒸馏指令对：
```bat
python multi_agent_nlp_project.py distill --distill-src data/synth.jsonl --distill-out data/distill_pairs.jsonl
```
3. LoRA 微调（CPU/GPU常规）：
```bat
python lora_distill.py --data data/distill_pairs.jsonl --model qwen/Qwen1.5-0.5B --output runs/qwen-mini-lora --epochs 1 --batch 2 --fp16
```
4. QLoRA (4bit, 需 GPU + bitsandbytes)：
```bat
python lora_distill.py --data data/distill_pairs.jsonl --model qwen/Qwen1.5-1.8B-Chat --output runs/qwen-lora-4bit --epochs 1 --batch 2 --qlora --fp16
```

### 2. 评估指标运行
```bat
python multi_agent_nlp_project.py eval --rounds 2 --requirements "严谨性,逻辑连贯" --report eval.json --html-report eval.html
```
控制台示例：
```
📈 评估汇总: {"len_gain_avg":0.132,"ttr_gain_avg":0.045,"repetition_delta_avg":0.021,"n":2,...}
```

### 3. 英文合成与蒸馏
```bat
python multi_agent_nlp_project.py synthesize --rounds 2 --lang en --requirements "academic polish,logical coherence" --out data/synth_en.jsonl
python multi_agent_nlp_project.py distill --distill-src data/synth_en.jsonl --distill-out data/distill_en.jsonl
```

### 4. 禁用组件做消融
```bat
python multi_agent_nlp_project.py demo --rounds 2 --no-tools --no-memory
```

### 5. 长文本文件优化 (新功能)
当你的学术初稿非常长（数千到数万字符），可以放到一个 `.txt` 文件中，使用 `--text-file` 自动分段迭代优化。

新增参数说明：
| 参数 | 默认 | 说明 |
|------|------|------|
| --text-file <path> | - | 指定要优化的长文本文件路径（UTF-8 编码） |
| --chunk-size <int> | 5000 | 单段最大字符数（简单按句子重组）|
| --chunk-overlap <int> | 200 | 相邻段落的尾部重叠字符数，提高连续性（首段无重叠）|
| --max-chunks <int> | 0 | 限制最多处理的段数（0 表示不限制，仅用于快速试验）|

使用示例：
```bat
python multi_agent_nlp_project.py demo --text-file paper_draft.txt --rounds 2 --requirements "学术表达提升,结构清晰" --report long.json --html-report long.html
```
限制处理前 3 段并调整分段大小：
```bat
python multi_agent_nlp_project.py demo --text-file paper_draft.txt --rounds 1 --chunk-size 3000 --chunk-overlap 150 --max-chunks 3 --requirements "学术表达提升;逻辑结构优化"
```
交互模式下同样适用（设置 ENABLE_INTERACTIVE=1）：
```bat
python multi_agent_nlp_project.py --text-file paper_draft.txt --rounds 2 --requirements "学术表达提升,逻辑结构优化"
```
输出 JSON 报告结构（`aggregated` 示例）：
```text
{
  "file": "paper_draft.txt",
  "chunks": 4,
  "chunk_size": 5000,
  "overlap": 200,
  "requirements": ["学术表达提升","结构清晰"],
  "final_text": "<所有优化后分段拼接>",
  "segments": [
    {
      "segment_index": 0,
      "original_length": 4987,
      "optimized_length": 5120,
      "final_segment_text": "...",
      "round_logs": [ { "round": 0, ... }, { "round": 1, ... } ]
    },
    {
      "segment_index": 1, ...
    }
  ]
}
```

注意事项：
- 分段是启发式按句号/英文标点拆分，不保证严格语义边界；可根据需要调低或调高 `--chunk-size`。
- `--chunk-overlap` 通过前一段尾部补偿上下文连续性，过大可能重复；过小可能导致跨段衔接生硬。
- 记忆检索在每个分段内部独立运行（默认 DummyEmbeddings 时效果有限）。如需跨段全局一致性，可后续再对 `final_text` 做一次整体精修。
- 评审 Agent B 的建议仅针对当前分段，不跨分段；可以在后处理中汇总所有 `priority_issues` 聚合全局改进点。
- 若文件非常大（> 数十万字符），建议先用外部工具粗清洗或分章，再交给本系统处理，减少调用成本。

下一步可扩展：
- 跨段全局汇总轮次（第二阶段整体协作）。
- 分段并行处理（当前顺序执行）。
- Diff 合并为段间全局摘要。

---
## 数据格式规范
### 合成数据 (synthesize 输出 JSONL，每行一个对象)
```text
{
  "id": "case_0",
  "input": "原始种子文本",
  "requirements": ["学术表达提升", "结构清晰"],
  "final": "最终优化文本",
  "log": [
    {"round": 0, "user_input": "...", "requirements": ["..."], "timestamp": "..."},
    {"round": 1, "optimized_text": "...", "agent_b_feedback": "...", "scores": {"quality": 8.0, "rigor": 7.0}, "diff": "...", ...},
    {"round": 2, ...}
  ],
  "created_at": "ISO 时间",
  "teacher_signal": "最后一轮 optimized_text (用于蒸馏)",
  "scores": {"quality": 8.0, "rigor": 7.0, "logic": 7.0, "novelty": 6.0}
}
```

### 蒸馏对 (distill 输出 JSONL)
```json
{"instruction": "优化以下学术段落，满足需求: 学术表达提升, 结构清晰\n原文: ...", "output": "教师信号文本", "scores": {"quality":8.0,"rigor":7.0}}
```

### LoRA 训练拼接格式
`lora_distill.py` 会将每条 `instruction/output` 拼为：
```
指令:\n<instruction>\n\n优质答案:\n<output>
```
用于常规自回归训练。

---
## 报告与可视化 (JSON / HTML)
- `--report <path>` 输出结构化 JSON：包含最终文本与全量日志。
- `--html-report <path>` 输出可视化报告：
  - 轮次文本
  - 评分徽章
  - Diff 展开/折叠 (新增绿色 / 删除红色)
  - 工具调用观察
  - 指标表格 (eval 模式汇总)

HTML 生成逻辑：`generate_html_report()`。

---
## 评估指标说明
| 指标 | 说明 |
|------|------|
| len_gain | 长度变化比例 (final_len - orig_len) / orig_len |
| ttr_gain | Type Token Ratio 提升 (词汇多样性) |
| repetition_delta | 前5高频 token 占比下降 (越大越少重复) |
| readability_gain | 可读性代理值提升 (基于句长反函数) |
| coherence_gain | 句间词汇交集/并集平均提升 (连贯性代理) |
| sent_var_delta | 原句长方差 - 新句长方差 (正值表示更均匀) |
| bigram_rep_delta | 前5高频二元组占比下降 (重复模式减少) |
| quality/rigor/logic/novelty | Agent B JSON 评分均值 (eval 汇总) |

其余指标与 README 旧版本保持一致含义。

---
## 回退策略与健壮性
| 层 | 主实现 | 回退1 | 回退2 |
|----|--------|-------|-------|
| LLM | ChatOpenAI | HTTPFallbackChat (直接POST) | DummyLLM |
| Embeddings | OpenAIEmbeddings | DummyEmbeddings (全零) | - |
| VectorStore | FAISS | SimpleVectorStore | - |
| Search | SerpAPIWrapper | 占位 stub | - |

触发条件：缺失 API Key / 初始化异常 → 打印 ⚠️ 或 ❌ 并降级。无 Key 仍可跑流程但内容无真实优化价值。

---
## 环境变量一览
| 名称 | 说明 | 必填 | 默认 |
|------|------|------|------|
| OPENAI_API_KEY | OpenAI/兼容接口 Key | 否 | - |
| SERPAPI_API_KEY | SerpAPI Key | 否 | - |
| OPENAI_BASE_URL | 接口 Base URL | 否 | https://api.chatanywhere.tech/v1 |
| LLM_MODEL | 模型名称 | 否 | gpt-4o-mini |
| EMBED_MODEL_NAME | Embedding 模型名 | 否 | text-embedding-3-small |
| ENABLE_INTERACTIVE | 交互模式开关 | 否 | 0 |

DeepSeek 自动规范：若 Base URL 包含 `deepseek.com` 且模型名非官方，自动切换为 `deepseek-chat` 或 `deepseek-reasoner`。

---
## 交互模式 (ENABLE_INTERACTIVE)
将 `.env` 中 `ENABLE_INTERACTIVE=1` 后：
- 运行脚本即进入单次协作流程（基于 `--text` 或默认初稿）。
- 可额外整合自定义后续交互（当前示例模式仅一次 collaborate 调用）。

---
## 性能与调优建议
| 目标 | 建议 |
|------|------|
| 加快速度 | 减少 `--rounds`，禁用工具/记忆 (`--no-tools --no-memory`)，缩短初稿长度 |
| 减少显存 | 使用 QLoRA 4bit (`--qlora`)，调低 `--max-length`，减小 batch |
| 提升多样性 | 在合成阶段增加不同需求组合与种子语料 |
| 教师质量 | 使用更强模型 (如 gpt-4 / 深度推理模型) 生成合成数据再蒸馏 |
| 减少指令截断 | 调整 `lora_distill.py --max-length` |
| 更平稳训练 | 增大 `--epochs`、设置 `--warmup-ratio`、调整 `--gradient-accum` |

---
## 常见问题 FAQ
1. 导入 faiss 报错 → 自动回退 SimpleVectorStore，可忽略或安装 `faiss-cpu`。
2. 搜索结果是占位 → 未配置 `SERPAPI_API_KEY`。
3. 输出始终很“空” → 使用的是 DummyLLM（缺少 OPENAI_API_KEY）。
4. Embeddings 404 → 兼容接口不支持嵌入，已使用 DummyEmbeddings 继续流程。
5. bitsandbytes 安装失败 → 不用 `--qlora`，只跑普通 LoRA。
6. 指令或回答过长 OOM → 减小 `--max-length` / 降低 batch / 使用梯度累积。
7. HTML 报告未生成（某命令）→ `synthesize` / `distill` 模式不生成流程 HTML。
8. 需要英文写作 → 设置 `--lang en` 并使用英文需求。
9. 评分 JSON 解析失败 → Agent B 输出未按规范格式，可能是模型响应质量不足；可提高模型质量或微调提示词。
10. Diff 太长 → 内部有截断逻辑 (400 行)；可以自行修改 `_compute_diff` 扩展。

---
## 扩展开发指南
### 1. 添加新工具
在 `multi_agent_nlp_project.py` 中 TOOLS 构建段添加：
```text
from langchain_core.tools import Tool
from your_lib import YourClient
client = YourClient(...)
new_tool = Tool(name="领域检索", func=client.query, description="输入关键词返回专业数据库摘要")
TOOLS.append(new_tool)
```
并在 `_plan_and_act()` 中根据需求关键字触发调用。

### 2. 新增指标
在 `evaluate()` 中：
1. 计算单案例新字段 → `case_record['your_metric'] = value`
2. 汇总部分加入平均值。
3. README 指标说明同步更新。

### 3. 增加第三个 Agent
- 新建模板 `agent_c_template`（如 引用核查 / 方法设计）。
- 在 `collaborate()` 循环中插入调用顺序，传入需要的上下文。
- 将其反馈写入 `memory.add_memory()` 以参与召回。

### 4. 改写输出结构
可将 Agent A 输出格式从“说明 + 正文”拆分为 JSON，便于后处理 diff 或分类：修改 `agent_a_template` 并适配 `_extract_section()`。

### 5. 替换 Embeddings 服务
将 OpenAIEmbeddings 替换为自托管（如 BGE / GTE）：在初始化段添加自定义 embedding 类并保持接口 `embed_query(embed_documents)`。

---
## 测试与验证
示例最小测试 (可选，文件 `tests/test_basic.py`)：
```python
from multi_agent_nlp_project import dual_agent_system

def test_dummy_flow():
    final, log = dual_agent_system.collaborate("测试初稿", ["学术表达提升"], rounds=1)
    assert isinstance(final, str)
    assert len(log) >= 2
```
运行：
```bat
python -m pytest -q
```
可再补充：
- 工具禁用分支测试 (`--no-tools`)
- 回退机制测试（不设置 Key）
- evaluate 输出结构测试

---
## Roadmap 后续扩展
- 更细粒度的工具触发策略（意图分类 / 内容分析）
- 结构化改动记录 (diff → 标注 JSON)
- 更丰富评估指标（语法错误率 / 引用一致性）
- 多 Agent (>2) 协同：方法设计 / 引用检索 / 校对角色
- `--no-tools` / `--no-memory` 消融实验报告自动汇总
- 导出对比报告 (HTML 聚合所有案例差异)
- 高质量教师模型 + 自动质量过滤阈值

---
## 许可与免责声明
目前脚本未附带明确开源许可。如需正式使用或二次分发，请补充 MIT / Apache-2.0 等协议文件。

本项目仅用于研究与教学演示：
- 输出质量依赖上游 LLM 能力与提示规范；
- 生成内容可能包含不准确信息，需人工校对；
- 请遵守所调用 API 的服务条款与用量限制。

---
## 快速命令速查 (Cheat Sheet)
```bat
# Demo + 报告
python multi_agent_nlp_project.py demo --rounds 2 --report demo.json --html-report demo.html

# 数据合成
python multi_agent_nlp_project.py synthesize --rounds 3 --requirements "学术表达提升,结构清晰,可读性增强" --out data/synth.jsonl

# 蒸馏对生成
python multi_agent_nlp_project.py distill --distill-src data/synth.jsonl --distill-out data/distill_pairs.jsonl

# 评估汇总
python multi_agent_nlp_project.py eval --rounds 2 --requirements "严谨性,逻辑连贯" --report eval.json --html-report eval.html

# LoRA 微调
python lora_distill.py --data data/distill_pairs.jsonl --model qwen/Qwen1.5-0.5B --output runs/qwen-mini-lora --epochs 1 --batch 2 --fp16

# QLoRA 微调
python lora_distill.py --data data/distill_pairs.jsonl --model qwen/Qwen1.5-1.8B-Chat --output runs/qwen-lora-4bit --epochs 1 --batch 2 --qlora --fp16
```

---

欢迎提出功能需求或进一步优化建议！
