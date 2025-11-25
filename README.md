# 多智能体学术写作优化系统（Academic Multi-Agent Writing Optimizer）

> 统一整合原 `README.md`、`PROJECT_STRUCTURE.md`、`GETTING_STARTED.md`、`web_interface/README.md` 内容，并做去重与结构化重写。本文件为唯一主 README，其余冗余说明文件已标记合并。全部内容中文说明，便于在科研与工程场景下快速理解与扩展。

---
## 目录
1. 项目概览与目标
2. 核心特性速览
3. 总体架构与模块关系图
4. 代码结构与文件职责详解
5. 环境与依赖（统一版 requirements）
6. 安装与快速启动（CLI / Web / Notebook）
7. 配置说明（.env / 运行时动态配置）
8. 使用模式详解：`demo` / `synthesize` / `eval` / `distill` / 长文本优化
9. 多智能体协作机制（Agent A 优化 + Agent B 审核）
10. 工具调用与回退策略（Search / Python REPL / 文件读写）
11. 向量记忆与检索策略（FAISS / 简易内存回退）
12. 高级学术评估指标体系（9 维 + 增强统计）
13. 数据管线：合成 → 蒸馏 → LoRA / QLoRA 微调
14. Web 图形界面与实时任务管理说明
15. 报告与可视化（HTML / JSON / 文本导出）
16. 测试与质量保障（回退设计 + 去耦合策略）
17. 常见问题 FAQ 与故障排查
18. 维护与扩展建议（可插拔 / 模块解耦）
19. 路线图与后续提升方向
20. 许可与引用

---
## 1. 项目概览与目标
本项目实现一个 **多智能体协作闭环学术写作优化系统**：
- Agent A（Optimizer）：针对草稿执行结构化、学术化、逻辑与表达提升。
- Agent B（Reviewer）：严谨评审并给出多维评分（quality/rigor/logic/novelty 等）与下一轮改进建议。
- 多轮交替后输出最终优化文本 + 详细协作日志（diff、评分、工具调用、记忆检索等）。
- 支持 **数据合成、蒸馏样本构造、LoRA/QLoRA 微调**，形成“生成 → 高质量 teacher signal → 微调 student”闭环。
- 集成 **高级学术文本质量评估体系（9 维指标）**，可比较优化前后改进幅度，支持权重自定义与场景化配置。
- 提供 **CLI、Web 图形界面与 Notebook** 三种使用方式；具备 **稳健回退与最小可运行模式（DummyLLM + 简易向量检索 + 工具 stub）**，保障在缺少 API Key 或外部依赖时仍能演示流程。

适用场景：论文段落打磨、方法/结果/讨论优化、学术表达风格统一、教育演示、多智能体策略实验、构建蒸馏训练数据、轻量模型学术写作能力增强。

---
## 2. 核心特性速览
- ✅ 多轮双 Agent 对抗协作（优化 + 审核 + 建议收敛）
- ✅ 学术高级评分体系（9 维 + 改进对比 + 自定义权重）
- ✅ 可回退的 LLM 初始化链（LangChain ChatOpenAI → HTTPFallbackChat → DummyLLM）
- ✅ 工具层 ReAct 风格触发（搜索 / Python REPL / 文件读写），可一键禁用
- ✅ 向量记忆（FAISS + Embeddings）→ 无依赖时自动回退为简单 Token 交集检索
- ✅ 长文本智能分段（句子边界 + 重叠）与按段优化汇总
- ✅ 数据全流程：`synthesize` → `distill` → `lora_distill.py` 微调 → `hf_student_llm.py` 加载本地学生模型
- ✅ 丰富 HTML 报告（Diff 高亮、评分卡片、进度与改进幅度可视化）
- ✅ Web 实时任务进度（Socket.IO 推送 + 多任务状态）
- ✅ 自适应中文/英文 + 支持混合文本
- ✅ 健壮回退：缺 Key / 缺模型 / 缺检索库 ≈ 仍能跑通基础流程

---
## 3. 总体架构与模块关系图（文字示意）
```
┌──────────────────────────────────────────────┐
│ multi_agent_nlp_project.py                  │
│  ├─ LLM 初始化与回退链 (OpenAI / HTTP / Dummy) │
│  ├─ 工具集合 (Search / Python REPL / File IO)  │
│  ├─ 向量记忆 (FAISS 或 SimpleVectorStore)      │
│  ├─ DualAgentAcademicSystem                  │
│  │   ├─ Agent A Prompt + Chain               │
│  │   ├─ Agent B Prompt + Chain               │
│  │   ├─ 协作日志记录 (scores/diff/tools/memory) │
│  │   ├─ evaluate / synthesize / distill      │
│  │   └─ HTML 报告生成                        │
│  ├─ 长文本分段 + 文件优化                     │
│  └─ CLI 入口 (demo / synthesize / eval / distill)│
├──────────────────────────────────────────────┤
│ metrics.py (9 维学术质量指标计算)              │
├——─────────────────────────────────────────────┤
│ demo_metrics.py (指标演示)                   │
├──────────────────────────────────────────────┤
│ lora_distill.py (LoRA / QLoRA 微调)           │
│ hf_student_llm.py (本地学生模型封装)           │
├──────────────────────────────────────────────┤
│ web_interface/ (Flask + SocketIO 实时界面)     │
│   ├─ app.py (API + 任务管理 + 报告下载)        │
│   ├─ start_web.py (启动脚本)                 │
│   ├─ index.html / static/ (UI & 前端逻辑)     │
└──────────────────────────────────────────────┘
```

---
## 4. 代码结构与文件职责详解
| 文件 | 职责 | 关键点 |
|------|------|--------|
| `multi_agent_nlp_project.py` | 核心框架与 CLI | 多智能体、工具、记忆、长文本、报告、数据加工 |
| `metrics.py` | 学术 9 维指标 + 综合评分 + 对比 | 零外部依赖，适配中英文，权重可自定义 |
| `demo_metrics.py` | 指标系统完整演示 | 单文本 / 前后对比 / 不同权重场景 / 字符条形图 |
| `lora_distill.py` | LoRA/QLoRA 微调 | 读取蒸馏对 JSONL，HF + PEFT + bitsandbytes 回退 |
| `hf_student_llm.py` | 本地学生模型封装 | .invoke 接口兼容多链式 | 
| `web_interface/app.py` | Web API + 任务线程 + SocketIO 推送 | 支持文本/文件优化、合成、评估、蒸馏、下载 |
| `web_interface/index.html` | 主页面 | 多标签 UI，进度、对比、配置、下载 |
| `web_interface/static/js/app.js` | 前端实时逻辑 | WebSocket 订阅、进度更新、表单提交 |
| `data/seeds.txt` | 种子语料示例 | 多场景多缺陷短句，用于合成实验 |
| `tests/test_flow.py` | 基础单元测试 | 回退解析、分段、需求解析正确性 |


---
## 5. 环境与依赖
统一后的 `requirements.txt` 已合并 Web 与核心依赖，去重并保留版本约束。核心分组：
- 基础与运行：`numpy`, `requests`, `python-dotenv`, `tiktoken`
- LangChain & OpenAI 兼容：`langchain*`, `google-search-results`
- 检索：`faiss-cpu`（可选，缺失则自动回退）
- 模型微调：`transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`（可选）
- Web 服务：`Flask`, `Flask-SocketIO`, `Flask-CORS`, `python-socketio`, `python-engineio`, `Werkzeug`
- 测试：`pytest`

> 若仅需本地最小演示（无网络/无 GPU），可以忽略安装 bitsandbytes、SerpAPI Key、FAISS；系统将自动使用 DummyLLM + 简易内存检索。

---
## 6. 安装与快速启动
### 6.1 创建虚拟环境（Windows CMD）
```bat
cd D:\Projects\NLP\multi_agent_NLP
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```
### 6.2 Linux / macOS
```bash
cd /path/to/multi_agent_NLP
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```
### 6.3 CLI 快速示例
```bat
python multi_agent_nlp_project.py demo --rounds 2 --text "这是一个需要提升学术表达与逻辑清晰度的段落。" --requirements "学术表达提升;逻辑结构优化" --html-report demo.html
```
### 6.4 Web 界面
```bat
cd web_interface
python start_web.py
:: 浏览器访问 http://localhost:5000
```
### 6.5 Notebook
```bash
jupyter notebook multi_agent_nlp_project.ipynb
```

---
## 7. 配置说明（.env）
`.env.example` 提供字段：
```ini
OPENAI_API_KEY=            # 可选；缺失则使用 DummyLLM
OPENAI_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-reasoner
SERPAPI_API_KEY=           # 可选；缺失则搜索工具返回占位信息
EMBED_MODEL_NAME=text-embedding-3-small
ENABLE_INTERACTIVE=0
```
说明：
- 未设置 `OPENAI_API_KEY`：仍能跑流程（占位输出），便于教学本地验证。
- DeepSeek Base URL 场景：自动模型名规范（chat / reasoner）。
- embeddings 当前在 DeepSeek 常无接口 → 自动回退 DummyEmbeddings 以启用记忆逻辑。

Web 端提供 `/api/config` 动态更新本次会话模型与 Key（不写回文件）。

---
## 8. 使用模式详解
| 命令 | 说明 | 常用参数 |
|------|------|----------|
| `demo` | 单文本 / 长文本（可分段）多轮优化 | `--text` / `--text-file` / `--rounds` / `--chunk-size` / `--html-report` |
| `synthesize` | 基于种子多轮协作合成数据集 | `--seeds-file` / `--rounds` / `--out` |
| `eval` | 多用例批量评估改进与多维统计 | `--rounds` / `--report` / `--html-report` |
| `distill` | 从合成数据构造指令蒸馏对 | `--distill-src` / `--distill-out` |

长文本参数：`--chunk-size`、`--chunk-overlap`、`--max-chunks` 控制分段与重叠。`--no-tools` 与 `--no-memory` 可做消融实验。

---
## 9. 多智能体协作机制
- Agent A Prompt：包含上一轮评分、记忆检索片段、工具观察、需求列表。
- Agent B Prompt：要求结构化输出 + JSON 评分段（quality/rigor/logic/novelty）。
- 日志字段`optimized_text`、`agent_b_feedback`、`scores`、`diff`、`tool_observations`、`timestamp`。
- 差异计算：统一 diff（截断 400 行）+ HTML 高亮新增 / 删除。

---
## 10. 工具调用与回退
触发条件：需求或文本含“search / 检索 / 事实 / 最新 / 引用” → 调用搜索；含 `python: ```python` 代码块 → 执行 Python REPL。
回退策略：
1. 未配置 SerpAPI → 返回占位提示不报错。
2. LangChain 相关包缺失 → 自动使用轻量 stub Tool 类 + PythonREPL stub，保持接口兼容。
3. 文件读写工具：简单 lambda（路径+内容）而非复杂依赖。

---
## 11. 向量记忆与检索策略
- 正常模式：`OpenAIEmbeddings` (或占位) + FAISS 向量相似度。
- 回退模式（依赖缺失）：对比 Token 集合 Jaccard 相似度排序。
- Memory 管理：写入（文本+元数据：时间戳 / 类型 / 轮数） → 查询时取 Top-k 相关段落辅助下一轮优化。

---
## 12. 高级学术评估指标体系（9 维）
| 指标 | 描述 | 典型提升途径 |
|------|------|--------------|
| 学术规范性 | 学术关键词、正式表达比例 | 替换口语化、补充术语与规范短语 |
| 引用与证据完整性 | 引用格式/证据性句子密度 | 添加数据/evidence 语句、规范引用标记 |
| 创新度 | 新颖性/对比结构/突破描述 | 加入“区别于/相比于/首次”等对比表达 |
| 语言流畅度 | 可读性与句式平滑 | 句子拆分、减少冗余、降低不必要复杂度 |
| 句子平衡 | 长短句分布与变异系数 | 合理拆分长句、合并过碎短句 |
| 论证强度 | 论点/证据/逻辑连接词覆盖 | 增补数据、使用逻辑衔接词、明确结论 |
| 表达多样性 | 词汇丰富度 + 起始句式多样 | 避免重复高频词，使用替换与多样开头 |
| 结构完整性 | 引言/主体/结论 + 限制与展望 | 添加“局限”、“未来工作”语句 |
| 时态一致性 | 中英文时态/时间指示协调 | 保持描述与结果时态统一，避免混乱 |

综合评分：默认权重（见 `metrics.py`），可自定义不同场景。支持优化前后改进对比与提升比率（`improvement_rate`）。

---
## 13. 数据管线：合成 → 蒸馏 → 微调
1. 合成 (`synthesize`)：多轮协作生成含 `teacher_signal` 的高质量终稿；保留轮次细节用于分析。
2. 蒸馏 (`distill`)：抽取指令 + 高质量输出对（JSONL），保留评分辅助过滤或加权。
3. 微调 (`lora_distill.py`)：加载基础 HF 模型 → 应用 LoRA / QLoRA → 训练 → 输出 `RUN_INFO.txt`。
4. 学生模型加载 (`hf_student_llm.py`)：作为 Agent A 本地模型，与远程高质量模型协作形成“学生 + 教师”混合系统。

---
## 14. Web 图形界面与实时任务
- 功能模块：文本优化 / 文件优化 / 数据合成 / 评估 / 蒸馏 / 配置管理。
- 实时性：SocketIO 推送 `task_update` / `round_update`。
- 下载：`/api/download/<task_id>/text|html|json` 提供最终文本、报告与结构化数据。
- API 摘要：
```
POST /api/optimize/text
POST /api/optimize/file
POST /api/synthesize
POST /api/evaluate
POST /api/distill
GET  /api/task/<task_id>
GET  /api/download/<task_id>/text|html|json
POST /api/config
```

---
## 15. 报告与可视化
- HTML 报告：包含最终文本、轮次盒、Diff 高亮、评分徽章、改进幅度卡片、基础与高级指标表格。
- JSON 报告：保存全部日志结构，便于后处理与统计分析。
- 文本导出：`--out-text-file` 写入最终优化结果（适合论文打磨流水线）。

---
## 16. 测试与质量保障
- 最小测试（`tests/test_flow.py`）：验证需求解析与长文本分段的关键边界。
- 回退逻辑：确保在缺少 LangChain 或 API Key 时不抛致命异常（使用 stub 工具 + DummyLLM）。
- 解耦策略：
  - 指标系统独立（不依赖外部模型）。
  - 微调脚本独立于主运行环境（仅在需要时安装训练依赖）。
  - Web 层与核心协作类通过稳定方法接口交互，不直接依赖内部实现细节。

---
## 17. 常见问题 FAQ
Q1: 没有 OPENAI_API_KEY 能运行吗？
A1: 可以，使用 DummyLLM 占位输出，流程与报告仍生成。
Q2: FAISS 安装失败？
A2: 自动回退为简单词集相似度检索；性能下降但功能可用。
Q3: 为什么搜索工具总是返回占位？
A3: 未设置 SERPAPI_API_KEY；只需在 .env 中补充即可启用真实搜索。
Q4: 报告未出现高级指标？
A4: 确认 `metrics.py` 正常导入；若被修改导致异常会跳过该部分。
Q5: 长文本分段不理想？
A5: 调整 `--chunk-size` 与 `--chunk-overlap`，或前处理去除异常超长句。
Q6: Web 端“连接失败”？
A6: 确认端口未被占用，或关闭防火墙，重新运行 `start_web.py`。
Q7: 两个 Agent 输出太相似？
A7: 可在后续扩展中为 Agent A / B 使用不同模型或不同系���提示强化差异化。

---
## 18. 维护与扩展建议
- 新增指标：在 `metrics.py` 中添加函数并纳入 `overall_quality_score` 权重。
- 新增工具：在工具初始化处追加 `Tool(...)`，并在 `_plan_and_act` 中增加触发条件。
- 多模型混合：通过 `hf_student_llm.py` 构建本地学生模型 + 远程教师模型，实现角色差异与蒸馏闭环。
- 评估增强：加入统计显著性（Bootstrap / t-test）对多轮改进进行显著性检验。
- 数据过滤：对蒸馏 pairs 根据综合评分或单维阈值做质量筛选。

---
## 19. 路线图
| 阶段 | 计划 | 价值 |
|------|------|------|
| 短期 | 增加英文完整示例与更多测试 | 稳定与多语言覆盖 |
| 中期 | 引入显著性检验与更细粒度因果链分析 | 提升学术严谨度 |
| 中期 | Web 增加可拖拽批量文件处理与进度汇总 | 提升生产力 |
| 长期 | 引入可学习的策略调度（强化学习选择工具/记忆） | 提升智能体自适应性 |
| 长期 | 模型对齐与安全过滤模块 | 面向生产与合规扩展 |

---
## 20. 许可与引用
- 本项目采用开放许可（请查阅根目录 LICENSE，如未包含可补充）。
```
作者. 多智能体学术写作优化框架: 协作、评审与蒸馏管线. 项目仓库, 2025.
```

---
## 🆕 混合模式 & 本地学生模型蒸馏说明
为充分体现知识蒸馏价值，系统支持 **Agent A 使用本地轻量 Qwen 学生模型**，Agent B 使用远程高质量教师模型（如 DeepSeek）。流程：
1. 数据合成：`synthesize` 生成包含 teacher_signal 的高质量优化终稿。
2. 蒸馏抽取：`distill` 从合成日志生成指令-输出 JSONL 对。
3. LoRA 微调：`lora_distill.py` 基于 Qwen 小模型和蒸馏数据进行 LoRA / QLoRA 训练（输出适配器目录）。
4. 混合模式：CLI 加 `--hybrid --student-base-model ... --student-lora-dir ...`，Agent A 加载学生模型(+LoRA)，Agent B 继续使用远程教师模型；产生更真实的“学生学习”对比。
5. 持久化配置：使用 `--student-save-config data/student_last_loaded.json` 保存当前学生配置。

### 环境变量
| 变量 | 作用 | 示例 |
|------|------|------|
| STUDENT_BASE_MODEL | 学生基础 HF 模型 | Qwen/Qwen1.5-1.8B-Chat |
| STUDENT_LORA_DIR | LoRA 适配器目录（训练输出） | runs/qwen-mini-lora |
| STUDENT_MAX_NEW_TOKENS | 生成上限 | 512 |
| FORCE_STUDENT_STUB | 设为 1 时使用占位学生模型（CI/无网络） | 0/1 |

### Stub 测试模式
设置 `FORCE_STUDENT_STUB=1` 时：
- 不下载 HuggingFace 权重；
- 学生模型输出为占位标记，结构完整；
- 适合单元测试和快速验证管线。

### 快速示例
```bat
:: 1. 合成高质量 teacher_signal 数据
python multi_agent_nlp_project.py synthesize --rounds 3 --seeds-file data\seeds.txt --out data\synth.jsonl

:: 2. 蒸馏抽取指令-输出对
python multi_agent_nlp_project.py distill --distill-src data\synth.jsonl --distill-out data\distill_pairs.jsonl

:: 3. LoRA 微调学生模型
python lora_distill.py --data data\distill_pairs.jsonl --model Qwen/Qwen1.5-1.8B-Chat --output runs\qwen-mini-lora --epochs 1 --batch 2 --fp16

:: 4. 启用混合模式进行真实协作优化
python multi_agent_nlp_project.py demo --hybrid --student-base-model Qwen/Qwen1.5-1.8B-Chat --student-lora-dir runs\qwen-mini-lora --rounds 2 --text "这是一段需要提升学术表达的示例文本。" --requirements "学术表达提升;逻辑结构优化" --html-report hybrid.html --student-save-config data\student_last_loaded.json
```
> 如果网络或资源受限，可临时设置 `FORCE_STUDENT_STUB=1`：
```bat
set FORCE_STUDENT_STUB=1
python multi_agent_nlp_project.py demo --hybrid --text "占位测试文本" --requirements "学术表达提升"
```
---
