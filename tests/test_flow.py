import os
os.environ['FORCE_STUDENT_STUB'] = '1'

import importlib
import json

# 导入主模块
proj = importlib.import_module('multi_agent_nlp_project')


def test_parse_requirements_basic():
    reqs = proj.parse_requirements('学术表达提升;逻辑结构优化,可读性增强；重复度下降', ['默认'])
    assert '学术表达提升' in reqs and '逻辑结构优化' in reqs
    assert '可读性增强' in reqs and '重复度下降' in reqs
    assert '默认' not in reqs  # 已被实际解析替换


def test_split_long_text_segments():
    # 构造一个包含多句的文本
    text = '句子一。句子二很长需要被分段以测试算法。句子三。' * 10
    chunks = proj._split_long_text(text, chunk_size=60, overlap=10)
    assert len(chunks) >= 3  # 应该被拆分
    # 检查重叠是否生效（除第一段外）
    if len(chunks) > 1:
        assert chunks[1][:10] == chunks[0][-10:]


def test_dummy_llm_fallback_collaborate():
    # 强制清除 OPENAI_API_KEY 触发 DummyLLM
    os.environ.pop('OPENAI_API_KEY', None)
    # 重新导入触发 init_llm 回退（简单策略：直接使用已初始化的 llm 若为 DummyLLM）
    llm = proj.llm
    # 若当前不是 DummyLLM，直接构造一个占位对象
    class _Dummy:
        model_name = 'dummy-llm'
        def invoke(self, prompt):
            return '[Dummy response]'
        def __call__(self, prompt):  # 使其变为 callable，便于 LangChain 包装为 Runnable
            return self.invoke(prompt)
        def __or__(self, other):
            return other
    if not getattr(llm, 'model_name', '').startswith('dummy'):
        llm = _Dummy()
    system = proj.DualAgentAcademicSystem(llm, [], proj.vectorstore, enable_tools=False, enable_memory=False)
    final_text, log = system.collaborate('这是一个测试占位回退流程的段落。', ['学术表达提升'], rounds=1)
    assert isinstance(final_text, str) and len(log) >= 2  # round 0 + round 1
    # 评分可能为空（Dummy），但日志结构应存在 diff 字段
    assert 'diff' in log[-1]


def test_generate_html_report_structure():
    # 构造最小日志
    log = [
        {"round": 0, "user_input": "原始", "requirements": ["学术表达提升"], "timestamp": "t0"},
        {"round": 1, "optimized_text": "优化后文本", "agent_b_feedback": "反馈", "scores": {"quality": 8}, "diff": "- 原始\n+ 优化后文本", "tool_observations": "(无)", "timestamp": "t1"}
    ]
    html = proj.generate_html_report('测试报告', '优化后文本', log, summary={"len_gain_avg": 0.1, "quality_avg": 7.5})
    assert '<html>' in html and '测试报告' in html
    assert '优化后文本' in html and '质量' in html or 'quality' in html
    assert 'diff' in html.lower()


def test_distill_pair_generation(tmp_path):
    # 创建一个合成数据文件
    synth_path = tmp_path / 'synth.jsonl'
    record = {
        'input': '原始文本',
        'requirements': ['学术表达提升'],
        'final': '最终优化文本',
        'teacher_signal': '最终优化文本',
        'scores': {"quality": 9}
    }
    synth_path.write_text(json.dumps(record, ensure_ascii=False) + '\n', encoding='utf-8')
    out_path = tmp_path / 'distill.jsonl'
    system = proj.DualAgentAcademicSystem(proj.llm, [], proj.vectorstore, enable_tools=False, enable_memory=False)
    system.prepare_distillation_pairs(synth_path, out_path)
    assert out_path.exists()
    content = out_path.read_text(encoding='utf-8').strip().splitlines()
    assert len(content) == 1 and 'instruction' in content[0] and 'output' in content[0]


def test_hybrid_student_stub_mode():
    """验证混合模式构建时 Agent A 使用学生 stub 或基础 Qwen 模型名称。"""
    os.environ['FORCE_STUDENT_STUB'] = '1'
    os.environ['STUDENT_BASE_MODEL'] = 'Qwen/Qwen1.5-1.8B-Chat'
    system = proj.build_hybrid_dual_agent_system()
    assert getattr(system.agent_a_llm, 'model_name', '') in ('student-stub', 'Qwen/Qwen1.5-1.8B-Chat')
    out = system.agent_a_llm.invoke({'round_num': 1, 'text': '混合模式测试'})
    assert isinstance(out, str) and len(out) > 0
