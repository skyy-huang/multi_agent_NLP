# 多智能体学术写作优化系统

> 双 Agent 协作 (优化 / 评审) + 简易 ReAct 工具使用 + 向量记忆 (FAISS 回退) + 数据合成与评估。

## 1. 项目概览
本项目演示一个用于学术写作优化的多智能体闭环：
- Agent A (Optimizer)：根据用户需求与工具观察改写文本；
- Agent B (Reviewer)：针对 Agent A 输出进行结构/表达/逻辑评审并提出下一轮建议；
- 多轮迭代，通过记忆与工具检索不断提升文本质量；
- 可批量“蒸馏”生成教学或训练用 JSONL 数据；
- 提供轻量评估指标（长度变化、TTR、多次重复度变化）。

适用于：原型验证 / Prompt 实验 / 数据合成；非生产环境，但结构可扩展。

## 2. 核心特性
- 多轮协作：`--rounds` 控制迭代次数；
- 工具层：网络搜索 (SerpAPI)，Python REPL，文件读写；
- 记忆层：优先使用 FAISS + Embeddings；如缺失依赖/无 API Key 回退到简易内存检索；
- LLM 回退链：`ChatOpenAI (langchain)` → `HTTPFallbackChat` → `DummyLLM`；
- 数据合成：多种种子文本批量运行写入 JSONL；
- 评估模块：代理式多轮并计算简单写作优化代理指标；
- CLI 增强：文本、需求、种子文件、输出路径、语言等可配置。

## 3. 目录结构
```
multi_agent_nlp_project.py      # 主脚本 (CLI + 系统实现)
multi_agent_nlp_project.ipynb  # 初始 Notebook 参考
requirements.txt               # 依赖列表
.env.example                   # 环境变量示例
data/                          # 合成数据输出目录 (执行后生成)
README.md                      # 使用与架构说明
```

## 4. 安装与环境准备 (Windows cmd)
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
```
根据需要在 `.env` 中设置：
```
OPENAI_API_KEY=你的Key
SERPAPI_API_KEY=可选，用于搜索
OPENAI_BASE_URL=https://api.deepseek.com   # 或其他兼容 OpenAI 接口
LLM_MODEL=deepseek-reasoner               # 或 deepseek-chat / gpt-4o-mini 等
EMBED_MODEL_NAME=text-embedding-3-small   # 若 base 支持 embeddings
ENABLE_INTERACTIVE=0                      # 置 1 开启交互模式
```

## 5. 快速开始示例
演示 (默认 2 轮)：
```bat
python multi_agent_nlp_project.py demo
```
指定文本与需求、轮次：
```bat
python multi_agent_nlp_project.py demo --rounds 3 --text "这是一段测试初稿，结构略显松散，需要提升专业性。" --requirements "学术表达提升;逻辑结构优化"
```
英文模式：
```bat
python multi_agent_nlp_project.py demo --lang en --requirements "academic polish,logical coherence"
```

## 6. CLI 用法
```text
python multi_agent_nlp_project.py [command] [--options]

command:
  demo        运行单条文本多轮优化
  synthesize  批量合成 JSONL 数据集
  eval        运行轻量评估 (内置测试用例)

通用参数:
  --rounds <int>           协作轮次 (>=1)
  --requirements <str>     逗号或分号分隔需求列表
  --lang zh|en             语言 (影响默认初稿/需求)

demo 专用:
  --text <str>             自定义初始文本

synthesize 专用:
  --seeds-file <path>      种子文本文件 (每行一条)，缺省用内置3条
  --out <path>             输出 JSONL 文件路径

eval 专用:
  (使用内置测试，无额外参数)
```

## 7. 数据合成 (synthesize)
默认会写入到 `data/synth_academic_<时间戳>.jsonl`：
```bat
python multi_agent_nlp_project.py synthesize --rounds 3 --requirements "学术表达提升,结构清晰,可读性增强"
```
输出 JSONL 每行字段：
```json
{
  "id": "case_0",
  "input": "原始种子文本",
  "requirements": ["学术表达提升", ...],
  "final": "最终优化文本",
  "log": [ { "round": 1, "optimized_text": "...", "agent_b_feedback": "..." }, ... ],
  "created_at": "ISO 时间戳"
}
```
自定义种子文件：
```
seeds.txt:
文本A...
文本B...
```
```bat
python multi_agent_nlp_project.py synthesize --seeds-file seeds.txt --out data/custom.jsonl
```

## 8. 评估 (eval)
运行：
```bat
python multi_agent_nlp_project.py eval --rounds 2 --requirements "严谨性,逻辑连贯"
```
控制台输出示例：
```
📈 评估汇总: {"len_gain_avg":0.132,"ttr_gain_avg":0.045,"repetition_delta_avg":0.021,"n":2}
```
指标说明：
- len_gain：长度变化比例 (final_len - orig_len) / orig_len；
- ttr_gain：类型-标记比 (Type Token Ratio) 提升；
- repetition_delta：前5高频 token 占比的降低（越大越好表示重复减少）。

## 9. 系统内部流程 (架构)
```
User Text + Requirements
          │
          ▼
     Agent A (Optimizer)  ←── 工具观察 (搜索 / REPL / 文件IO)
          │ 输出优化稿 + 修改说明
          ▼
     Agent B (Reviewer)
          │ 反馈评审/问题/下轮建议
          ▼
   MemoryManager (写入轮次文本与反馈)
          │ 召回相似片段辅助下一轮
          └──> 多轮循环直至 rounds 完成
```
关键模块：
- DualAgentAcademicSystem.collaborate(): 多轮主循环；
- _plan_and_act(): 基于需求/文本简单触发工具；
- MemoryManager: 抽象向量存储与回退实现；
- synthesize_dataset(): 种子批量运行并序列化；
- evaluate(): 运行多轮并计算代理指标。

## 10. 回退与健壮性策略
| 层 | 主路径 | 回退1 | 回退2 |
|----|--------|-------|-------|
| LLM | ChatOpenAI | HTTPFallbackChat (POST) | DummyLLM |
| Embeddings | OpenAIEmbeddings | DummyEmbeddings (全零) | - |
| VectorStore | FAISS | SimpleVectorStore | - |
| Search Tool | SerpAPIWrapper | 占位 stub | - |

触发条件：
- 缺失 API Key / 初始化异常 → 打印 ⚠️ 或 ❌ 并使用下一级回退。

## 11. 环境变量清单
| 名称 | 说明 | 必填 | 默认 |
|------|------|------|------|
| OPENAI_API_KEY | OpenAI/兼容接口 Key | 否 | - |
| SERPAPI_API_KEY | SerpAPI Key | 否 | - |
| OPENAI_BASE_URL | OpenAI 兼容接口 Base URL | 否 | https://api.chatanywhere.tech/v1 |
| LLM_MODEL | 模型名称 | 否 | gpt-4o-mini |
| EMBED_MODEL_NAME | Embedding 模型名称 | 否 | text-embedding-3-small |
| ENABLE_INTERACTIVE | 交互模式开关 | 否 | 0 |

## 12. 常见问题 (FAQ)
1. Unable to import faiss / vectorstores → 移除或跳过 `faiss-cpu`，项目会自动退化为 SimpleVectorStore。
2. 搜索结果总是占位 → 未设置 `SERPAPI_API_KEY`。
3. 输出英文而非中文 → 检查 `--lang` 参数或默认模型输出偏好。
4. Embeddings 报错 404 → DeepSeek 等兼容接口无嵌入服务，自动使用 DummyEmbeddings。
5. DummyLLM 输出不优化 → 需配置真实 LLM Key 才会产生高质量改写。

## 13. 后续扩展建议 (Roadmap)
- 更细粒度的工具触发策略 (意图分类 / 内容分析)。
- 记录结构化修改操作 (diff / 打标段落变化)。
- 引入更丰富的评估指标 (语法错误率, 逻辑一致性打分)。
- 支持多 Agent (>2) 协同角色：方法设计 / 引用检索 / 校对。
- 增加 `--no-tools` / `--no-memory` 用于消融实验。
- 导出对比报告 (HTML) 聚合所有轮次差异。

## 14. 最小测试 (可选)
如需加入测试，可创建 `tests/test_basic.py`：
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

## 15. 许可证
此演示脚本未附带明确许可证，若需开源发布请补充 MIT / Apache-2.0 等声明。

---

欢迎提出功能需求或进一步优化建议！
