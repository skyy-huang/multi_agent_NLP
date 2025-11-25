#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
演示脚本：展示新的学术指标系统的使用
"""

import sys
import io
from pathlib import Path

# 设置stdout编码为UTF-8以支持Unicode字符
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from metrics import AcademicMetrics, quick_evaluate


def demo_single_text_evaluation():
    """演示1：单个文本的完整评估"""
    print("\n" + "="*70)
    print("演示1: 单个文本的完整学术质量评估")
    print("="*70)
    
    sample_text = """
    本研究基于深度学习方法，采用多层神经网络架构进行模型设计。我们收集了大规模的标注数据集，
    包含来自不同来源的1000+样本。实验结果表明，所提方法在标准测试集上达到了最先进的性能，
    准确率相比现有方法提升了15%。进一步的消融实验验证了各关键模块的有效性。
    
    虽然取得了显著成果，但本工作仍存在以下局限：(1)数据规模有限，(2)计算效率需改进。
    未来我们计划使用更大规模的数据进行训练，并探索模型压缩技术以提高实际应用的可行性。
    """
    
    quick_evaluate(sample_text)


def demo_before_after_comparison():
    """演示2：优化前后的对比分析"""
    print("\n" + "="*70)
    print("演示2: 优化前后的对比分析")
    print("="*70)
    
    original_text = """
    我们做了一个关于文本优化的研究。我们收集了很多数据。我们用这些数据训练了模型。
    模型的效果不错。我们的方法比其他方法好。还有一些问题需要解决。今后要继续改进。
    """
    
    optimized_text = """
    本研究提出了一套基于多智能体协作的文本优化框架。首先，我们收集并标注了大规模学术文本数据集，
    包含来自多个学科领域的近5000篇论文样本。其次，我们设计了由优化Agent和评审Agent组成的协作系统，
    通过多轮迭代逐步改进文本质量。实验结果表明，本方法显著提升了文本的学术规范性、逻辑连贯性和表达流畅度，
    平均改进幅度达到12%以上，超越现有文本优化基线方法。
    
    尽管取得了积极进展，本工作仍存在若干局限性：(1)当前仅针对中文学术文本进行优化，
    (2)模型的可解释性有待深入研究，(3)计算效率需进一步提升。
    未来的研究方向主要包括：拓展至多语言支持、增强模型的透明性与鲁棒性、以及在实际学术写作系统中的应用验证。
    """
    
    print("\n【原始文本统计】")
    original_eval = AcademicMetrics.overall_quality_score(original_text)
    print(f"字数: {original_eval['word_count']}, 句数: {original_eval['sentence_count']}, 段数: {original_eval['paragraph_count']}")
    print(f"综合评分: {original_eval['overall_score']:.4f}")
    
    print("\n【优化文本统计】")
    optimized_eval = AcademicMetrics.overall_quality_score(optimized_text)
    print(f"字数: {optimized_eval['word_count']}, 句数: {optimized_eval['sentence_count']}, 段数: {optimized_eval['paragraph_count']}")
    print(f"综合评分: {optimized_eval['overall_score']:.4f}")
    
    # 对比分析
    comparison = AcademicMetrics.compare_improvements(original_text, optimized_text)
    
    print("\n【各维度改进情况】")
    print(f"{'维度':<25} {'原始分':<12} {'优化分':<12} {'改进幅度':<12}")
    print("-" * 61)
    for metric in original_eval['scores'].keys():
        orig = original_eval['scores'][metric]
        opt = optimized_eval['scores'][metric]
        improvement = opt - orig
        color_icon = "[+]" if improvement > 0 else "[-]" if improvement < 0 else "[=]"
        print(f"{metric:<25} {orig:<12.4f} {opt:<12.4f} {color_icon} {improvement:+.4f}")
    
    print(f"\n{'总体质量':<25} {original_eval['overall_score']:<12.4f} {optimized_eval['overall_score']:<12.4f} {comparison['overall_improvement']:+.4f}")


def demo_custom_weights():
    """演示3：自定义权重的评估"""
    print("\n" + "="*70)
    print("演示3: 自定义权重的评估")
    print("="*70)
    
    text = """
    本文提出了一个基于神经网络的创新算法。与传统方法相比，我们的方法不仅在理论上有所突破，
    而且在实际应用中也取得了显著的改进。大量的实验数据证明了本方法的有效性。
    """
    
    # 权重1: 强调学术规范性和引用完整性（适合学位论文）
    academic_weights = {
        'academic_formality': 0.25,
        'citation_completeness': 0.20,
        'novelty': 0.08,
        'language_fluency': 0.12,
        'sentence_balance': 0.08,
        'argumentation': 0.12,
        'expression_diversity': 0.05,
        'structure_completeness': 0.08,
        'tense_consistency': 0.02
    }
    
    # 权重2: 强调创新和论证（适合研究论文）
    research_weights = {
        'academic_formality': 0.12,
        'citation_completeness': 0.12,
        'novelty': 0.20,
        'language_fluency': 0.10,
        'sentence_balance': 0.08,
        'argumentation': 0.20,
        'expression_diversity': 0.08,
        'structure_completeness': 0.08,
        'tense_consistency': 0.02
    }
    
    # 权重3: 强调易读性和多样性（适合科普文章）
    popular_weights = {
        'academic_formality': 0.05,
        'citation_completeness': 0.08,
        'novelty': 0.10,
        'language_fluency': 0.25,
        'sentence_balance': 0.15,
        'argumentation': 0.10,
        'expression_diversity': 0.15,
        'structure_completeness': 0.08,
        'tense_consistency': 0.04
    }
    
    academic_result = AcademicMetrics.overall_quality_score(text, weights=academic_weights)
    research_result = AcademicMetrics.overall_quality_score(text, weights=research_weights)
    popular_result = AcademicMetrics.overall_quality_score(text, weights=popular_weights)
    
    print("\n同一文本在不同评估标准下的得分:")
    print(f"  学位论文标准(强调规范和引用): {academic_result['overall_score']:.4f}")
    print(f"  研究论文标准(强调创新和论证): {research_result['overall_score']:.4f}")
    print(f"  科普文章标准(强调易读和多样): {popular_result['overall_score']:.4f}")
    
    print("\n权重差异导致的评分变化:")
    print(f"  学位论文 vs 研究论文: {academic_result['overall_score'] - research_result['overall_score']:+.4f}")
    print(f"  研究论文 vs 科普文章: {research_result['overall_score'] - popular_result['overall_score']:+.4f}")


def demo_detailed_metrics():
    """演示4: 详细的各维度指标展示"""
    print("\n" + "="*70)
    print("演示4: 详细的各维度指标展示")
    print("="*70)
    
    text = """
    随着人工智能技术的快速发展，自然语言处理在学术写作领域的应用前景日益广阔。
    本文系统分析了当前基于深度学习的文本优化方法，并提出了一个创新的多智能体协作框架。
    通过大量实验证明，本方法在文本质量、学术规范性和创新性等多个维度均取得显著改进。
    """
    
    result = AcademicMetrics.overall_quality_score(text)
    
    metric_names = {
        'academic_formality': '学术规范性',
        'citation_completeness': '引用完整性',
        'novelty': '创新度',
        'language_fluency': '语言流畅度',
        'sentence_balance': '句子平衡度',
        'argumentation': '论证强度',
        'expression_diversity': '表达多样性',
        'structure_completeness': '结构完整性',
        'tense_consistency': '时态一致性'
    }
    
    print("\n各维度详细评分:")
    print(f"{'维度':<20} {'得分':<10} {'等级':<15} {'可视化':<25}")
    print("-" * 70)
    
    for metric, name in metric_names.items():
        score = result['scores'][metric]
        
        # 确定等级
        if score >= 0.8:
            level = "优秀 ***"
        elif score >= 0.6:
            level = "良好 **"
        elif score >= 0.4:
            level = "一般 *"
        else:
            level = "需改进"
        
        # 可视化条形图
        bar_length = int(score * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        print(f"{name:<20} {score:<10.4f} {level:<15} [{bar}]")
    
    print("\n" + "-" * 70)
    print(f"综合评分: {result['overall_score']:.4f} ", end="")
    
    overall = result['overall_score']
    if overall >= 0.75:
        print("(优秀)")
    elif overall >= 0.60:
        print("(良好)")
    elif overall >= 0.45:
        print("(中等)")
    else:
        print("(需改进)")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("    学术文本优化评价指标系统 - 演示脚本")
    print("=" * 70)
    
    # 运行所有演示
    demo_single_text_evaluation()
    demo_before_after_comparison()
    demo_custom_weights()
    demo_detailed_metrics()
    
    print("\n" + "="*70)
    print("OK 所有演示完成!")
    print("="*70)
    print("\n更多用法:")
    print("  1. 在评估模式中使用: python multi_agent_nlp_project.py eval --html-report report.html")
    print("  2. 在 Python 代码中导入: from metrics import AcademicMetrics")
    print("  3. 查看详细文档: cat README.md 中的第11-13章节")
    print()
