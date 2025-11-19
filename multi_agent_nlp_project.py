"""
Multi-agent academic writing optimization demo script.
Features:
- Dual agents (Optimizer A / Reviewer B) performing multi-round improvement.
- Minimal ReAct-style tool planning (web search / Python REPL / file IO).
- Vector memory via FAISS (fallback to simple in-memory similarity if deps missing).
- Dataset synthesis (JSONL) + lightweight evaluation metrics.
- Robust fallback chain: ChatOpenAI → HTTP compatible client → DummyLLM.
- Enhanced CLI parameters: rounds, custom text, requirements list, seeds file, output path, language.
Run: python multi_agent_nlp_project.py demo --rounds 3 --requirements "学术表达提升,逻辑结构优化" --text "待优化初稿..."
"""
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv

# New stdlib imports
import json
from pathlib import Path
import argparse
import re
import requests

# ---------------------------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
IS_DEEPSEEK = "deepseek.com" in (OPENAI_BASE_URL or "").lower()
# Normalize DeepSeek model naming per official docs
if IS_DEEPSEEK:
    lm_lower = (LLM_MODEL or "").lower()
    if lm_lower not in ("deepseek-chat", "deepseek-reasoner"):
        new_model = "deepseek-reasoner" if ("reason" in lm_lower or "think" in lm_lower) else "deepseek-chat"
        print(f"ℹ️ 检测到 DeepSeek 且模型名 '{LLM_MODEL}' 非官方推荐，已自动规范为 '{new_model}'。")
        LLM_MODEL = new_model

# ---------------------------------------------------------------------------
# 2. Safe LLM initialization (falls back to dummy LLM if API key missing)
# ---------------------------------------------------------------------------
# Remove eager import of langchain_openai to avoid ImportError in Dummy path

class DummyLLM:
    """Fallback LLM used when API keys are missing; mimics .invoke interface."""
    def __init__(self):
        self.model_name = "dummy-llm"
    def invoke(self, prompt: Dict | str):
        if isinstance(prompt, dict):
            return f"[DummyLLM response for keys: {list(prompt.keys())}]"
        return "[DummyLLM generic response]"
    def __or__(self, other):  # allow chaining compatibility
        return other


class HTTPFallbackChat:
    """直接使用 OpenAI 兼容接口的简单回退客户端。满足 .invoke(dict|str) 接口。"""
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        # 兼容有/无 /v1 前缀
        if self.base_url.endswith('/v1'):
            self.endpoint = f"{self.base_url}/chat/completions"
        else:
            self.endpoint = f"{self.base_url}/v1/chat/completions"
    def invoke(self, prompt: Dict | str):
        if isinstance(prompt, dict):
            # 将字典内容拼合为 user 消 Messages
            user_content = '\n'.join(f"{k}: {v}" for k, v in prompt.items())
        else:
            user_content = str(prompt)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an academic writing optimization assistant."},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        try:
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                return f"[HTTPFallbackChat Error {resp.status_code}: {resp.text[:200]}]"
            data = resp.json()
            # OpenAI 兼容格式：choices[0].message.content
            return data.get("choices", [{}])[0].get("message", {}).get("content", "[No content]")
        except Exception as e:
            return f"[HTTPFallbackChat Exception: {e}]"
    def __or__(self, other):
        return other


def init_llm():
    if not OPENAI_API_KEY:
        print("⚠️ OPENAI_API_KEY missing; using DummyLLM.")
        return DummyLLM()
    # 优先尝试 langchain ChatOpenAI
    try:
        from langchain_openai import ChatOpenAI  # lazy import
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            api_key=(lambda: OPENAI_API_KEY),
            base_url=OPENAI_BASE_URL,
        )
        print(f"✅ Primary LLM ({LLM_MODEL}) initialized via {OPENAI_BASE_URL}.")
        return llm
    except Exception as e:
        print(f"❌ Failed to init ChatOpenAI: {e}; attempting HTTP fallback.")
        try:
            fallback_llm = HTTPFallbackChat(OPENAI_BASE_URL, OPENAI_API_KEY, LLM_MODEL)
            # 简单探针调用确认可用
            probe = fallback_llm.invoke("probe")
            if probe.startswith("[HTTPFallbackChat Error") or probe.startswith("[HTTPFallbackChat Exception"):
                print("⚠️ HTTP fallback probe failed; using DummyLLM.")
                return DummyLLM()
            print(f"✅ HTTP Fallback LLM ({LLM_MODEL}) ready via {OPENAI_BASE_URL}.")
            return fallback_llm
        except Exception as e2:
            print(f"❌ Fallback initialization failed: {e2}; using DummyLLM.")
            return DummyLLM()


llm = init_llm()

# ---------------------------------------------------------------------------
# 3. Tools setup
# ---------------------------------------------------------------------------
try:
    from langchain_core.tools import Tool
    from langchain_experimental.utilities import PythonREPL
    from langchain_community.utilities import SerpAPIWrapper
except ImportError as e:
    raise RuntimeError(f"Missing langchain packages: {e}. Run pip install -r requirements.txt")

# SerpAPI tool
TOOLS: List[Tool] = []  # add explicit type for clarity
if SERPAPI_API_KEY:
    search_wrapper = SerpAPIWrapper(search_engine="google", serpapi_api_key=SERPAPI_API_KEY)
    search_tool = Tool(
        name="网络搜索",
        func=search_wrapper.run,
        description="实时信息查询：输入搜索关键词"
    )
    TOOLS.append(search_tool)
else:
    def _search_stub(q: str) -> str:
        return f"[SerpAPI 未配置，无法执行搜索: {q}]"
    search_tool = Tool(name="网络搜索", func=_search_stub, description="SerpAPI 未配置占位工具")
    TOOLS.append(search_tool)

# Python REPL tool
python_repl = PythonREPL()
python_repl_tool = Tool(
    name="Python REPL",
    func=python_repl.run,
    description="执行格式正确的 Python 代码"
)
TOOLS.extend([
    python_repl_tool,
    Tool(name="读取文件", func=lambda fn: read_file(fn), description="读取文件内容。输入为文件名"),
    Tool(name="写入文件", func=lambda arg: write_file(arg), description="写入文件内容。格式: filename.txt,内容")
])
print(f"🔧 已加载 {len(TOOLS)} 个工具")

# ---------------------------------------------------------------------------
# 4. Vector store (FAISS) + embeddings
# ---------------------------------------------------------------------------
USE_FAISS = True
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore
    import faiss
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings as LCEmbeddings
except ImportError as e:
    # Fallback to simple in-memory store without external deps
    print(f"⚠️ 向量存储依赖缺失: {e}. 使用简易内存检索代替 FAISS。")
    USE_FAISS = False
    FAISS = None  # type: ignore
    InMemoryDocstore = None  # type: ignore
    faiss = None  # type: ignore
    try:
        from langchain_core.documents import Document  # may still be available
    except Exception:
        class Document:  # minimal fallback
            def __init__(self, page_content: str, metadata: Optional[Dict] = None):
                self.page_content = page_content
                self.metadata = metadata or {}


EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "text-embedding-3-small")  # 1536 dims
EMBED_DIM = 1536  # text-embedding-3-small dimension

# Provide a consistent DummyEmbeddings implementing LangChain interface and callable fallback
class DummyEmbeddings:
    def embed_query(self, t: str):
        return [0.0] * EMBED_DIM
    def embed_documents(self, docs: List[str]):
        return [[0.0] * EMBED_DIM for _ in docs]
    def __call__(self, t: str):  # some FAISS versions call embedding_function(query)
        return self.embed_query(t)

if OPENAI_API_KEY:
    try:
        if IS_DEEPSEEK:
            # DeepSeek 的 OpenAI 兼容接口通常不提供 embeddings；直接使用占位向量避免 404
            print("ℹ️ 检测到 DeepSeek base_url，跳过 OpenAIEmbeddings，使用占位嵌入以启用记忆功能。")
            embeddings_model = DummyEmbeddings()
        else:
            from langchain_openai import OpenAIEmbeddings  # lazy import
            embeddings_model = OpenAIEmbeddings(
                model=EMBED_MODEL_NAME,
                api_key=(lambda: OPENAI_API_KEY),
                base_url=OPENAI_BASE_URL
            )
    except Exception as e:
        print(f"❌ Embeddings 初始化失败: {e}; 使用占位 embedding 函数。")
        embeddings_model = DummyEmbeddings()
else:
    embeddings_model = DummyEmbeddings()
    print("⚠️ OPENAI_API_KEY 缺失，向量嵌入使用占位向量。")

# Simple tokenize for fallback similarity
_DEF_WORD_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")

def _simple_tokenize(text: str) -> List[str]:
    return _DEF_WORD_RE.findall(text or "")

if USE_FAISS:
    class EmbeddingAdapter(LCEmbeddings):
        """Embeddings adapter implementing LangChain Embeddings interface to silence deprecation warnings."""
        def __init__(self, base):
            self.base = base
        def embed_query(self, x: str) -> List[float]:
            try:
                return self.base.embed_query(x)
            except Exception:
                return [0.0] * EMBED_DIM
        def embed_documents(self, xs: List[str]) -> List[List[float]]:
            try:
                return self.base.embed_documents(xs)
            except Exception:
                return [[0.0] * EMBED_DIM for _ in xs]
    adapter = EmbeddingAdapter(embeddings_model)
    index = faiss.IndexFlatL2(EMBED_DIM)
    vectorstore = FAISS(adapter, index, InMemoryDocstore({}), {})
    print("🧠 向量数据库(FAISS)初始化完成")
else:
    class SimpleVectorStore:
        def __init__(self):
            self.docs: List[Document] = []
        def add_documents(self, docs: List[Document]):
            self.docs.extend(docs)
        def similarity_search(self, query: str, k: int = 3) -> List[Document]:
            q_tokens = set(_simple_tokenize(query))
            def score(doc: Document):
                d_tokens = set(_simple_tokenize(doc.page_content))
                if not q_tokens or not d_tokens:
                    return 0.0
                return len(q_tokens & d_tokens) / len(q_tokens | d_tokens)
            ranked = sorted(self.docs, key=score, reverse=True)
            return ranked[:k]
    vectorstore = SimpleVectorStore()
    print("🧠 向量数据库简化版初始化完成(无FAISS)")

# Simple memory manager around vector store (FAISS or fallback)
class MemoryManager:
    def __init__(self, vs, namespace: str = "global"):
        self.vs = vs
        self.namespace = namespace
        self._counter = 0
    def add_memory(self, text: str, metadata: Optional[Dict] = None):
        try:
            meta = metadata or {}
            meta.update({"namespace": self.namespace, "ts": datetime.now().isoformat()})
            doc = Document(page_content=text, metadata=meta)
            self.vs.add_documents([doc])
            self._counter += 1
        except Exception as e:
            print(f"⚠️ 写入记忆失败: {e}")
    def recall(self, query: str, k: int = 3) -> List[str]:
        try:
            res = self.vs.similarity_search(query, k=k)
            return [d.page_content for d in res]
        except Exception as e:
            print(f"⚠️ 读取记忆失败: {e}")
            return []

# ---------------------------------------------------------------------------
# 5. Dual Agent System
# ---------------------------------------------------------------------------
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class DualAgentAcademicSystem:
    def __init__(self, llm, tools, vectorstore):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.vectorstore = vectorstore
        self.memory = MemoryManager(vectorstore)
        self.collaboration_log: List[Dict] = []
        self._setup_agents()

    def _setup_agents(self):
        self.agent_a_template = PromptTemplate.from_template(
            """你是Agent A - 学术表达优化专家。\n轮次: 第{round_num}轮\n用户需求: {user_requirements}\n长程记忆检索片段:\n{memory_snippets}\n工具观察:\n{tool_observations}\n待优化文本:\n{text_to_optimize}\n{previous_feedback}\n请输出:\n**优化版本：**\n[优化后的完整文本]\n\n**修改说明：**\n[说明本轮修改要点]"""
        )
        self.agent_b_template = PromptTemplate.from_template(
            """你是Agent B - 学术评审专家。\n轮次: 第{round_num}轮\n用户需求: {user_requirements}\n优化文本:\n{optimized_text}\n请评审并输出:\n**本轮改进评价：**\n[...]\n\n**剩余主要问题：**\n[...]\n\n**下轮重点建议：**\n1. [...]\n2. [...]\n\n**改进优先级：**\n[...]"""
        )
        self.agent_a_chain = self.agent_a_template | self.llm | StrOutputParser()
        self.agent_b_chain = self.agent_b_template | self.llm | StrOutputParser()

    @staticmethod
    def _extract_section(text: str, start_token: str, end_token: str) -> str:
        lines = text.split('\n')
        collecting = False
        buf = []
        for l in lines:
            if start_token in l:
                collecting = True
                continue
            if collecting and end_token in l:
                break
            if collecting:
                buf.append(l)
        return '\n'.join(buf).strip()

    def _plan_and_act(self, text: str, requirements: List[str]) -> str:
        """A minimal ReAct-style tool use: decide to use search or python based on simple cues.
        Returns observation strings to feed into Agent A.
        """
        observations = []
        joined_req = ' '.join(requirements).lower()
        # If requirements indicate factual validation or search
        if any(kw in joined_req for kw in ["search", "检索", "事实", "最新", "引用"]):
            # naive query: take last sentence or keywords in quotes
            m = re.findall(r'"([^\"]+)"', text)
            query = m[-1] if m else text.split('。')[-1].strip() or text
            try:
                obs = self.tools["网络搜索"].run(query)
            except Exception as e:
                obs = f"[搜索异常: {e}]"
            observations.append(f"搜索[{query}] -> {obs[:300]}")
        # If code execution is requested (e.g., 'python:' prefix)
        code_blocks = re.findall(r'python:\s*```python\n([\s\S]*?)```', text, flags=re.IGNORECASE)
        for code in code_blocks[:1]:  # run at most one for safety
            try:
                out = self.tools["Python REPL"].run(code)
            except Exception as e:
                out = f"[代码执行异常: {e}]"
            observations.append(f"执行Python -> 输出: {str(out)[:200]}")
        return "\n".join(observations) if observations else "(无)"

    def collaborate(self, user_text: str, user_requirements: List[str], language: str = "中文", rounds: int = 3) -> Tuple[str, List[Dict]]:
        self.collaboration_log = [{"round": 0, "user_input": user_text, "requirements": user_requirements, "timestamp": datetime.now().isoformat()}]
        current_text = user_text
        previous_feedback = ""
        # Prime memory with the initial user input
        self.memory.add_memory(user_text, {"type": "user_input"})
        for r in range(1, rounds + 1):
            # recall top-k memory relevant to current text
            mem_snippets = self.memory.recall(current_text, k=3)
            tool_obs = self._plan_and_act(current_text, user_requirements)
            a_input = {
                "round_num": r,
                "text_to_optimize": current_text,
                "user_requirements": ', '.join(user_requirements),
                "previous_feedback": previous_feedback,
                "memory_snippets": '\n'.join(mem_snippets) if mem_snippets else "(无)",
                "tool_observations": tool_obs,
            }
            a_resp = self.agent_a_chain.invoke(a_input)
            optimized_text = self._extract_section(a_resp, "**优化版本：**", "**修改说明：**") or current_text

            b_input = {
                "round_num": r,
                "optimized_text": optimized_text,
                "user_requirements": ', '.join(user_requirements)
            }
            b_resp = self.agent_b_chain.invoke(b_input)

            # write memory
            self.memory.add_memory(optimized_text, {"type": "optimized_text", "round": r})
            self.memory.add_memory(b_resp, {"type": "feedback", "round": r})

            self.collaboration_log.append({
                "round": r,
                "agent_a_response": a_resp,
                "optimized_text": optimized_text,
                "agent_b_feedback": b_resp,
                "tool_observations": tool_obs,
                "timestamp": datetime.now().isoformat()
            })
            previous_feedback = b_resp
            current_text = optimized_text
            print(f"✅ Round {r} 完成")
            time.sleep(0.2)
        return current_text, self.collaboration_log

    # ----------------------------- Data Synthesis -----------------------------
    def synthesize_dataset(self, seeds: List[str], requirements: List[str], rounds: int = 3, out_path: Optional[Path] = None) -> Path:
        """Run collaborate() on multiple seeds and write JSONL dataset for distillation."""
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        if out_path is None:
            out_path = data_dir / f"synth_academic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for i, seed in enumerate(seeds):
                final_text, log = self.collaborate(seed, requirements, rounds=rounds)
                record = {
                    "id": f"case_{i}",
                    "input": seed,
                    "requirements": requirements,
                    "final": final_text,
                    "log": log,
                    "created_at": datetime.now().isoformat(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        print(f"📀 合成数据已写入: {out_path} (共 {count} 条)")
        return out_path

    # ----------------------------- Evaluation -----------------------------
    @staticmethod
    def _tokenize_zh(text: str) -> List[str]:
        # naive Chinese tokenization by characters and Latin words
        words = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text)
        return words

    def evaluate(self, cases: List[Tuple[str, List[str]]], rounds: int = 2) -> Dict:
        """Evaluate improvements with simple proxy metrics.
        cases: List of (text, requirements)
        Returns summary metrics and per-case results.
        """
        results = []
        for idx, (text, reqs) in enumerate(cases):
            final_text, _ = self.collaborate(text, reqs, rounds=rounds)
            # Metrics: length gain, type-token ratio, repetition reduction
            w0 = self._tokenize_zh(text)
            w1 = self._tokenize_zh(final_text)
            len_gain = (len(w1) - len(w0)) / max(1, len(w0))
            ttr0 = len(set(w0)) / max(1, len(w0))
            ttr1 = len(set(w1)) / max(1, len(w1))
            # repetition: share of top-5 most frequent tokens
            from collections import Counter
            c0 = Counter(w0)
            c1 = Counter(w1)
            rep0 = sum(x for _, x in c0.most_common(5)) / max(1, len(w0))
            rep1 = sum(x for _, x in c1.most_common(5)) / max(1, len(w1))
            results.append({
                "id": idx,
                "len_gain": round(len_gain, 3),
                "ttr_gain": round(ttr1 - ttr0, 3),
                "repetition_delta": round(rep0 - rep1, 3),
                "orig_len": len(w0),
                "final_len": len(w1),
            })
        # aggregate
        if results:
            avg = {
                "len_gain_avg": round(sum(r["len_gain"] for r in results)/len(results), 3),
                "ttr_gain_avg": round(sum(r["ttr_gain"] for r in results)/len(results), 3),
                "repetition_delta_avg": round(sum(r["repetition_delta"] for r in results)/len(results), 3),
                "n": len(results)
            }
        else:
            avg = {"len_gain_avg": 0.0, "ttr_gain_avg": 0.0, "repetition_delta_avg": 0.0, "n": 0}
        report = {"summary": avg, "cases": results}
        print("📈 评估汇总:", json.dumps(report["summary"], ensure_ascii=False))
        return report


dual_agent_system = DualAgentAcademicSystem(llm, TOOLS, vectorstore)
print("🤖 双Agent系统初始化完成")

# ---------------------------------------------------------------------------
# 6. Minimal CLI interaction and utilities
# ---------------------------------------------------------------------------
ENABLE_INTERACTIVE = os.getenv("ENABLE_INTERACTIVE", "0") == "1"
# --------------------- CLI helpers (enhanced) ---------------------

def parse_requirements(raw: Optional[str], default: List[str]) -> List[str]:
    if not raw:
        return default
    items = [x.strip() for x in re.split(r'[;,]', raw) if x.strip()]
    return items or default

def load_seeds_from_file(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception as e:
        print(f"⚠️ 种子文件读取失败: {e}")
        return None

def run_demo(rounds: int, text: Optional[str], requirements_raw: Optional[str], lang: str):
    base_default = (
        "This is a preliminary draft about multi-agent collaboration in academic writing." if lang == "en" else "这是一段关于多智能体协作进行学术写作优化的初稿。"
    )
    sample_text = text or base_default
    requirements = parse_requirements(requirements_raw, ["学术表达提升", "逻辑结构优化"] if lang == "zh" else ["academic polish", "logical coherence"])
    final_text, _ = dual_agent_system.collaborate(sample_text, requirements, rounds=rounds)
    print("\n📌 Final optimized text:\n", final_text)


def run_synthesis(rounds: int, requirements_raw: Optional[str], seeds_file: Optional[str], out: Optional[str]):
    default_seeds = [
        "本研究探讨了基于多智能体的文本优化框架，初步实验尚不充分。",
        "我们提出一个简单的管线，但方法部分缺乏清晰的因果论证。",
        "实验结果显示一定改进，但统计显著性需要进一步说明。",
    ]
    seeds = load_seeds_from_file(seeds_file) or default_seeds
    reqs = parse_requirements(requirements_raw, ["学术表达提升", "结构清晰", "可读性增强"])
    out_path = Path(out) if out else None
    dual_agent_system.synthesize_dataset(seeds, reqs, rounds=rounds, out_path=out_path)


def run_eval(rounds: int, requirements_raw: Optional[str]):
    tests = [
        ("本文提出一种方法，但存在一些问题，需要更严谨的叙述。", parse_requirements(requirements_raw, ["严谨性", "逻辑连贯"])),
        ("我们的实验结果较为有限，缺少消融实验。", parse_requirements(requirements_raw, ["补充实验建议", "学术化表达"])),
    ]
    dual_agent_system.evaluate(tests, rounds=rounds)


def build_arg_parser():
    p = argparse.ArgumentParser(description="Dual-agent academic optimizer")
    p.add_argument("command", nargs="?", default="demo", choices=["demo", "synthesize", "eval"], help="运行模式")
    p.add_argument("--rounds", type=int, default=2, help="协作轮次")
    p.add_argument("--text", type=str, help="自定义初始文本(仅 demo)")
    p.add_argument("--requirements", type=str, help="逗号/分号分隔的需求列表")
    p.add_argument("--seeds-file", type=str, help="种子文本文件路径(用于 synthesize)")
    p.add_argument("--out", type=str, help="输出 JSONL 文件路径(用于 synthesize)")
    p.add_argument("--lang", type=str, choices=["zh", "en"], default="zh", help="语言：zh 或 en")
    return p


if __name__ == "__main__":
    print("📦 环境检查: OPENAI_API_KEY={} SERPAPI_API_KEY={}".format(bool(OPENAI_API_KEY), bool(SERPAPI_API_KEY)))
    parser = build_arg_parser()
    args = parser.parse_args()
    rounds = max(1, args.rounds)
    if ENABLE_INTERACTIVE:
        print("🚀 交互模式开启 (设置 ENABLE_INTERACTIVE=1 关闭脚本自动演示)")
        run_demo(rounds, args.text, args.requirements, args.lang)
    else:
        if args.command == "demo":
            print("🚀 运行演示 (设置 ENABLE_INTERACTIVE=1 切换为交互模式)")
            run_demo(rounds, args.text, args.requirements, args.lang)
        elif args.command == "synthesize":
            print("🧪 运行数据合成")
            run_synthesis(rounds, args.requirements, args.seeds_file, args.out)
        elif args.command == "eval":
            print("🧮 运行评估")
            run_eval(rounds, args.requirements)
