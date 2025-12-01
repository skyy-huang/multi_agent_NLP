"""
学术文本优化评价指标计算模块
包含多维度的学术写作质量评估指标
"""

import re
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter
import statistics


class AcademicMetrics:
    """学术文本优化评价指标计算类"""
    
    @staticmethod
    def tokenize_zh(text: str) -> List[str]:
        """中文分词（简单的字符和英文单词级别分词）"""
        words = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text or "")
        return words
    
    @staticmethod
    def get_sentences(text: str) -> List[str]:
        """分句（支持中英文句号、问号、感叹号）"""
        sentences = [s.strip() for s in re.split(r'[。.!?！？]\s*', text) if s.strip()]
        return sentences
    
    @staticmethod
    def get_paragraphs(text: str) -> List[str]:
        """分段"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return paragraphs
    
    # ============ 1. 学术规范性指标 ============
    @staticmethod
    def academic_formality_score(text: str) -> float:
        """
        学术规范性评分 (0-1)
        检查是否使用了学术性词汇和短语
        """
        academic_keywords = [
            '研究', '方法', '结果', '论文', '分析', '实验', '数据', '模型', '算法', '证明',
            '假设', '理论', '框架', '体系', '机制', '性质', '特征', '现象', '规律', '原理',
            '基于', '通过', '采用', '提出', '发现', '表明', '证实', '验证', '评估', '改进',
            '显著', '明显', '进一步', '深入', '系统', '全面', '综合', '详细', '严谨', '科学'
        ]
        
        if not text:
            return 0.0
        
        words = AcademicMetrics.tokenize_zh(text)
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # 统计学术词汇出现次数
        academic_count = sum(1 for w in words if any(kw in w for kw in academic_keywords))
        
        # 基础分：学术词汇比例
        base_score = min(academic_count / max(1, total_words * 0.2), 1.0)
        
        # 奖励分：存在特定学术短语
        academic_phrases = ['本文', '研究表明', '根据', '因此', '综上所述', '综合来看']
        phrase_bonus = 0.1 if any(phrase in text for phrase in academic_phrases) else 0.0
        
        # 惩罚分：存在非学术表述
        informal_markers = ['就是', '其实', '哈哈', '呃', '吧', '啦', '呢']
        penalty = 0.15 if any(marker in text for marker in informal_markers) else 0.0
        
        score = min(max(base_score + phrase_bonus - penalty, 0.0), 1.0)
        return round(score, 4)
    
    # ============ 2. 引用与证据完整性 ============
    @staticmethod
    def citation_completeness_score(text: str) -> float:
        """
        引用完整性评分 (0-1)
        检查引用格式、引用密度、参考资料风格等
        """
        # 检测各种引用格式
        citation_patterns = [
            r'\[[\d,\-]+\]',  # [1], [1-3], [1,2,3]
            r'\([A-Z][a-z]+\s+\d{4}\)',  # (Author 2020)
            r'et\s+al\.?',  # et al.
            r'cite|reference|from\s+the',  # 引用动词
            r'引用|参考文献|参\s*[考參]|见\s*\[',  # 中文引用标记
        ]
        
        # 统计引用
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        sentences = AcademicMetrics.get_sentences(text)
        total_sentences = len(sentences)
        
        if total_sentences == 0:
            return 0.0
        
        # 引用密度（每5句一个引用为理想）
        ideal_citation_rate = total_sentences / 5.0
        citation_density = min(citation_count / max(1, ideal_citation_rate), 1.0)
        
        # 检查论据陈述（"数据表明", "研究显示", "结果表明" 等）
        evidence_phrases = ['数据', '结果', '发现', '表明', '证明', '验证', '实验', '调查']
        evidence_count = sum(1 for s in sentences if any(ph in s for ph in evidence_phrases))
        evidence_ratio = min(evidence_count / max(1, len(sentences) * 0.3), 1.0)
        
        score = (citation_density * 0.5 + evidence_ratio * 0.5)
        return round(min(score, 1.0), 4)
    
    # ============ 3. 创新度检测 ============
    @staticmethod
    def novelty_score(text: str) -> float:
        """
        创新度评分 (0-1)
        检测是否包含新颖的表述、观点或方法论述
        """
        novelty_markers = [
            '首次', '首先', '创新', '新颖', '独特', '突破', '颠覆', '革命性',
            '未见', '鲜有', '罕见', '引入', '发明', '创造', '提出新的',
            '不同以往', '区别于', '突破传统', '打破', '超越',
            '改进', '优化', '增强', '增加', '扩展', '完善', '推广'
        ]
        
        if not text:
            return 0.0
        
        # 统计创新标记词
        novelty_count = sum(1 for marker in novelty_markers if marker in text)
        
        # 检测对比结构（"以往...本文..." 等）
        contrast_patterns = [
            r'以往|传统|现有|现在',
            r'本文|本研究|本项目|本工作'
        ]
        has_contrast = all(any(re.search(p, text) for p in [cp]) for cp in contrast_patterns)
        
        # 基础分
        base_score = min(novelty_count * 0.08, 0.7)
        
        # 对比结构奖励
        contrast_bonus = 0.2 if has_contrast else 0.0
        
        score = min(base_score + contrast_bonus, 1.0)
        return round(score, 4)
    
    # ============ 4. 语言流畅度（Flesch 阅读难度改进版） ============
    @staticmethod
    def language_fluency_score(text: str) -> float:
        """
        语言流畅度评分 (0-1)
        基于改进的 Flesch 阅读等级指数
        """
        words = AcademicMetrics.tokenize_zh(text)
        sentences = AcademicMetrics.get_sentences(text)
        
        if len(words) == 0 or len(sentences) == 0:
            return 0.0
        
        # 计算中文改进版可读性指数
        word_count = len(words)
        sentence_count = len(sentences)
        
        # 检测长单词（4字以上汉字或较长英文）
        long_words = [w for w in words if len(w) >= 4]
        long_word_ratio = len(long_words) / max(1, len(words))
        
        # 改进的中文阅读难度公式
        # Flesch 中文改版 = 100 - (9.5 * long_words/words + 100 * sentences/words)
        avg_word_per_sent = word_count / max(1, sentence_count)
        readability_index = 100 - (9.5 * long_word_ratio + 60 * (1 / max(1, avg_word_per_sent)))
        
        # 标准化到 [0, 1]，假设范围是 [0, 100]
        fluency = max(0.0, min(readability_index / 100.0, 1.0))
        
        return round(fluency, 4)
    
    # ============ 5. 句子复杂度平衡度 ============
    @staticmethod
    def sentence_complexity_balance(text: str) -> float:
        """
        句子复杂度平衡度评分 (0-1)
        检测句子长度分布是否合理（不应该全是长句或全是短句）
        """
        sentences = AcademicMetrics.get_sentences(text)
        
        if len(sentences) < 2:
            return 0.0 if len(sentences) == 0 else 1.0
        
        # 计算每句的字数
        sent_lengths = [len(AcademicMetrics.tokenize_zh(s)) for s in sentences]
        
        # 理想的句长应该在 15-40 字之间（中文学术文本）
        ideal_range = (15, 40)
        
        # 计算在理想范围内的句子比例
        in_ideal = sum(1 for l in sent_lengths if ideal_range[0] <= l <= ideal_range[1])
        ideal_ratio = in_ideal / len(sent_lengths)
        
        # 计算句长的变异系数（变异越小，单调；变异越大，跳变）
        if len(sent_lengths) > 1:
            mean_len = statistics.mean(sent_lengths)
            variance = statistics.pvariance(sent_lengths)
            std_dev = math.sqrt(variance) if variance > 0 else 0
            cv = std_dev / mean_len if mean_len > 0 else 0
            
            # 理想的变异系数在 0.3-0.6 之间
            cv_score = 1 - abs(cv - 0.45) / 0.45 if cv > 0 else 0.5
            cv_score = max(0, min(cv_score, 1.0))
        else:
            cv_score = 1.0
        
        # 综合评分
        score = ideal_ratio * 0.6 + cv_score * 0.4
        return round(score, 4)
    
    # ============ 6. 论证强度评分 ============
    @staticmethod
    def argumentation_strength(text: str) -> float:
        """
        论证强度评分 (0-1)
        检测是否包含充分的论证要素（前提、论据、结论）
        """
        # 论点标记词
        thesis_markers = ['认为', '主张', '观点', '核心', '目标', '宗旨', '主题', '论点']
        
        # 论据标记词
        evidence_markers = ['数据', '案例', '例如', '事例', '实验', '调查', '结果', '事实', '证据']
        
        # 逻辑关联词
        logic_markers = ['因此', '所以', '由此', '由此可见', '进而', '从而', '既然', '假如', '如果', '当',
                        '而且', '并且', '同时', '另一方面', '相比之下', '不同于', '相反地', '总之', '综上所述']
        
        # 反驳/对比标记
        contrast_markers = ['但是', '然而', '虽然', '尽管', '不过', '反而', '相对地', '相比之下']
        
        sentences = AcademicMetrics.get_sentences(text)
        
        if len(sentences) == 0:
            return 0.0
        
        # 检测各要素的出现
        thesis_count = sum(1 for marker in thesis_markers if marker in text)
        evidence_count = sum(1 for sent in sentences if any(marker in sent for marker in evidence_markers))
        logic_count = sum(1 for sent in sentences if any(marker in sent for marker in logic_markers))
        contrast_count = sum(1 for marker in contrast_markers if marker in text)
        
        # 评估论证完整性
        thesis_score = min(thesis_count / max(1, len(sentences) * 0.2), 1.0)
        evidence_score = min(evidence_count / max(1, len(sentences) * 0.3), 1.0)
        logic_score = min(logic_count / max(1, len(sentences) * 0.4), 1.0)
        contrast_score = min(contrast_count / 2, 0.3)  # 对比元素不宜过多
        
        # 综合论证强度
        score = (thesis_score * 0.25 + evidence_score * 0.35 + logic_score * 0.3 + contrast_score * 0.1)
        return round(min(score, 1.0), 4)
    
    # ============ 7. 表达多样性 ============
    @staticmethod
    def expression_diversity(text: str) -> float:
        """
        表达多样性评分 (0-1)
        检测词汇丰富度、句式多样性
        """
        words = AcademicMetrics.tokenize_zh(text)
        sentences = AcademicMetrics.get_sentences(text)
        
        if len(words) == 0:
            return 0.0
        
        # 1. 词汇多样性 (TTR - Type Token Ratio)
        unique_words = len(set(words))
        ttr = unique_words / max(1, len(words))
        
        # 2. 句式多样性 - 检测不同的句子开头
        sentence_starts = [AcademicMetrics.tokenize_zh(s)[0] if AcademicMetrics.tokenize_zh(s) else "" 
                          for s in sentences]
        unique_starts = len(set(s for s in sentence_starts if s))
        sent_diversity = unique_starts / max(1, len(sentences)) if sentences else 0
        
        # 3. 避免词汇重复（检测最频繁词的占比）
        if words:
            word_freq = Counter(words)
            top_5_freq = sum(count for word, count in word_freq.most_common(5)) / len(words)
            repetition_score = 1 - top_5_freq  # 高频词占比越小越好
        else:
            repetition_score = 0.0
        
        # 综合表达多样性
        score = (ttr * 0.4 + sent_diversity * 0.3 + repetition_score * 0.3)
        return round(min(score, 1.0), 4)
    
    # ============ 8. 结构完整性评分 ============
    @staticmethod
    def structure_completeness(text: str) -> float:
        """
        结构完整性评分 (0-1)
        检测是否包含引言、主体、结论等关键部分的标记
        """
        # 各部分的标记词
        intro_markers = ['引言', '介绍', '背景', '现状', '意义', '问题']
        body_markers = ['方法', '分析', '论证', '实现', '设计', '过程', '步骤']
        conclusion_markers = ['结论', '总结', '综述', '综合', '最后', '总体来看', '综上所述']
        limitation_markers = ['局限', '不足', '缺陷', '限制', '改进']
        future_markers = ['未来', '后续', '展望', '进一步', '下一步']
        
        # 检测各部分
        has_intro = any(marker in text for marker in intro_markers)
        has_body = any(marker in text for marker in body_markers)
        has_conclusion = any(marker in text for marker in conclusion_markers)
        has_limitation = any(marker in text for marker in limitation_markers)
        has_future = any(marker in text for marker in future_markers)
        
        # 基础结构（至少应有引言、主体、结论）
        base_score = sum([has_intro, has_body, has_conclusion]) / 3.0
        
        # 高阶结构（包含局限和前景讨论）
        advanced_bonus = (has_limitation + has_future) * 0.1
        
        # 段落数量（学术文本通常分为多段）
        paragraphs = AcademicMetrics.get_paragraphs(text)
        paragraph_bonus = min(len(paragraphs) / 5.0, 0.15) if len(paragraphs) > 1 else 0.0
        
        score = min(base_score + advanced_bonus + paragraph_bonus, 1.0)
        return round(score, 4)
    
    # ============ 9. 时态一致性 ============
    @staticmethod
    def tense_consistency(text: str) -> float:
        """
        时态一致性评分 (0-1)
        检测过去式、现在式的一致性（主要针对英文）
        """
        # 过去式标记（English）
        past_tense = ['was', 'were', 'did', 'had', 'went', 'found', 'showed', 'demonstrated']
        # 现在式标记（English）
        present_tense = ['is', 'are', 'does', 'do', 'have', 'shows', 'demonstrates']
        
        # 中文时态标记
        past_markers = ['曾', '曾经', '过去', '已', '已经', '完成', '实现']
        present_markers = ['现在', '当前', '目前', '正在', '继续']
        
        words_lower = text.lower()
        
        # 英文时态检测
        past_count = sum(1 for marker in past_tense if f' {marker} ' in words_lower or f' {marker}.' in words_lower)
        present_count = sum(1 for marker in present_tense if f' {marker} ' in words_lower or f' {marker}.' in words_lower)
        
        # 中文时态检测
        past_count += sum(1 for marker in past_markers if marker in text)
        present_count += sum(1 for marker in present_markers if marker in text)
        
        # 如果没有明显的时态标记，则评分为1（自然一致）
        if past_count == 0 and present_count == 0:
            return 1.0
        
        # 计算时态的平衡性
        total = past_count + present_count
        if total == 0:
            return 1.0
        
        # 理想状态是时态用法不混乱（某一个占绝对优势或完全一致）
        ratio = max(past_count, present_count) / total
        consistency = ratio if ratio >= 0.8 else 1 - (ratio - 0.5) * 2
        
        return round(min(max(consistency, 0.0), 1.0), 4)
    
    # ============ 10. 整体质量综合评分 ============
    @staticmethod
    def overall_quality_score(text: str, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        整体质量综合评分
        计算所有指标并返回带权重的综合分数
        """
        # 默认权重
        if weights is None:
            weights = {
                'academic_formality': 0.15,
                'citation_completeness': 0.12,
                'novelty': 0.10,
                'language_fluency': 0.15,
                'sentence_balance': 0.10,
                'argumentation': 0.15,
                'expression_diversity': 0.08,
                'structure_completeness': 0.10,
                'tense_consistency': 0.05
            }
        
        # 计算各项指标
        scores = {
            'academic_formality': AcademicMetrics.academic_formality_score(text),
            'citation_completeness': AcademicMetrics.citation_completeness_score(text),
            'novelty': AcademicMetrics.novelty_score(text),
            'language_fluency': AcademicMetrics.language_fluency_score(text),
            'sentence_balance': AcademicMetrics.sentence_complexity_balance(text),
            'argumentation': AcademicMetrics.argumentation_strength(text),
            'expression_diversity': AcademicMetrics.expression_diversity(text),
            'structure_completeness': AcademicMetrics.structure_completeness(text),
            'tense_consistency': AcademicMetrics.tense_consistency(text),
        }
        
        # 计算加权综合分
        weighted_sum = sum(scores[k] * weights.get(k, 0) for k in scores.keys())
        total_weight = sum(weights.values())
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return {
            'scores': scores,
            'weights': weights,
            'overall_score': round(overall_score, 4),
            'text_length': len(text),
            'word_count': len(AcademicMetrics.tokenize_zh(text)),
            'sentence_count': len(AcademicMetrics.get_sentences(text)),
            'paragraph_count': len(AcademicMetrics.get_paragraphs(text))
        }
    
    # ============ 11. 文本改进对比分析 ============
    @staticmethod
    def compare_improvements(original_text: str, optimized_text: str) -> Dict[str, any]:
        """
        对比优化前后的改进情况
        """
        original_scores = AcademicMetrics.overall_quality_score(original_text)
        optimized_scores = AcademicMetrics.overall_quality_score(optimized_text)
        
        # 计算各维度的改进幅度
        improvements = {}
        for metric in original_scores['scores'].keys():
            orig = original_scores['scores'][metric]
            opt = optimized_scores['scores'][metric]
            improvement = opt - orig
            improvements[metric] = round(improvement, 4)
        
        # 计算整体改进
        overall_improvement = optimized_scores['overall_score'] - original_scores['overall_score']
        
        return {
            'original_scores': original_scores,
            'optimized_scores': optimized_scores,
            'metric_improvements': improvements,
            'overall_improvement': round(overall_improvement, 4),
            'improvement_rate': round((optimized_scores['overall_score'] - original_scores['overall_score']) / 
                                     max(0.1, original_scores['overall_score']), 4)
        }


# ============ 便捷函数 ============
def quick_evaluate(text: str) -> None:
    """快速评估单个文本，打印所有指标"""
    result = AcademicMetrics.overall_quality_score(text)
    
    print("=" * 60)
    print("学术文本质量评估报告")
    print("=" * 60)
    print(f"\n文本统计:")
    print(f"  字数: {result['word_count']}")
    print(f"  句数: {result['sentence_count']}")
    print(f"  段数: {result['paragraph_count']}")
    
    print(f"\n各维度评分 (0-1):")
    for metric, score in result['scores'].items():
        metric_name = metric.replace('_', ' ').title()
        bar = '█' * int(score * 20) + '░' * (20 - int(score * 20))
        print(f"  {metric_name:25s} {score:.4f} [{bar}]")
    
    print(f"\n整体综合评分: {result['overall_score']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    # 测试用例
    test_text = """
    本研究基于深度学习方法，采用多层神经网络架构进行模型设计。我们收集了大规模的标注数据集，
    包含来自不同来源的1000+样本。实验结果表明，所提方法在标准测试集上达到了最先进的性能，
    准确率相比现有方法提升了15%。进一步的消融实验验证了各关键模块的有效性。
    
    虽然取得了显著成果，但本工作仍存在以下局限：(1)数据规模有限，(2)计算效率需改进。
    未来我们计划使用更大规模的数据进行训练，并探索模型压缩技术以提高实际应用的可行性。
    """
    
    quick_evaluate(test_text)
